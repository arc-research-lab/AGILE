#include <linux/module.h>
#include <linux/slab.h>         
#include <linux/fs.h>           
#include <linux/cdev.h>
#include <linux/device.h>
#include <asm/io.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <linux/dmaengine.h>
#include <linux/delay.h>
#include <linux/list.h>
#include <linux/eventfd.h>
#include <linux/atomic.h>
#include <linux/spinlock.h>
#include <linux/ktime.h>

#include "../common/agile_kernel_driver.h"
#include "../common/agile_dma_cmd.h"

#define ALLOC_KMALLOC 1
#define DEVICE_NAME "AGILE-kernel"
#define CLASS_NAME  "AGILE Kernel Class"

#define ENABLE_DMA 1

struct dma_callback_param {
    struct dma_queue_pair * queue_pair;
    struct completion *event;
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t size;
    uint32_t identifier;
    uint64_t issue_start;
    uint64_t issue_finish;
};

struct dma_chan_list {
    struct dma_chan *chan;
    struct list_head list;
};

struct dma_chan_node {
    struct dma_chan *chan;
};

struct dma_queue_pair {
    uint32_t sq_head;
    uint32_t sq_tail;
    uint32_t cq_head;
    uint32_t cq_tail;
    uint32_t depth;
    uint32_t id;

    /**
     * The cmd queue won't be shared across threads
     */
    struct agile_dma_cmd_t *cmds; // located in CPU memory, written by GPU, read by CPU

    /**
     * The cpls may be used in different threads (callback functions) need use spinlock to protect. 
     */
    struct agile_dma_cpl_t *cpls; // located in GPU memory, written by CPU, read by GPU
    
    struct completion *events;

    struct dma_callback_param *cb_params; 

    /**
     * when update cpls in the callback function, need to use spinlock to protect
     */
    spinlock_t cpl_lock;
};

struct file_var {
    struct dma_buffer target_buffer;
    uint32_t cpu_dma_queue_num;
    uint32_t g_idx; // global index for allocating CPU DMA queues
    struct dma_queue_pair *cpu_dma_queues;
    uint64_t dram_offsets;
    uint64_t hbm_offsets;
};

static LIST_HEAD(chan_head);

static dev_t dev_num;

static dev_t dev_num;
static struct cdev  kernel_cdev;
static struct class *kernel_class;

static uint32_t total_dma_channels = 0;
static struct dma_chan_node * dma_channels;

static int fp_open(struct inode *inodep, struct file *filep) {
#if ENABLE_DMA
    dma_cap_mask_t mask;
    struct dma_chan *chan;
    bool chan_available = true;
    struct dma_chan_list *new_node;
    struct dma_chan_list *curr_node, *tmp;
    struct file_var *vars;
    int i = 0;

    
    vars = kzalloc(sizeof(struct file_var), GFP_KERNEL);
    vars->cpu_dma_queue_num = 0;
    vars->g_idx = 0;
    filep->private_data = vars;
    if (!filep->private_data) {
        return -ENOMEM;
    }
    pr_info("%s: Device opened\n", DEVICE_NAME);

    
    new_node = vmalloc(sizeof(struct dma_chan_list));
    if (!new_node) {
        pr_err("%s: Failed to allocate memory for new channel node\n", DEVICE_NAME);
        return -ENOMEM;
    }
    INIT_LIST_HEAD(&new_node->list);
    dma_cap_zero(mask);
    dma_cap_set(DMA_MEMCPY, mask);
    chan = dma_request_channel(mask, NULL, NULL);
    if (!chan) {
        chan_available = false;
        pr_err("%s: No DMA channel available\n", DEVICE_NAME);
        return -ENODEV;
    } else {
        total_dma_channels++;
        pr_info("%s: Allocated DMA channel %d: %s\n", DEVICE_NAME, total_dma_channels, dev_name(chan->device->dev));
    }
    new_node->chan = chan;
    list_add_tail(&new_node->list, &chan_head);

    while(chan_available){
        dma_cap_zero(mask);
        dma_cap_set(DMA_MEMCPY, mask);
        chan = dma_request_channel(mask, NULL, NULL);
        if (!chan) {
            chan_available = false;
            break;
        }
        // pr_info("%s: Allocated DMA channel %d: %s\n", DEVICE_NAME, total_dma_channels, dev_name(chan->device->dev));
        new_node = vmalloc(sizeof(struct dma_chan_list));
        if (!new_node) {
            pr_err("%s: Failed to allocate memory for new channel node\n", DEVICE_NAME);
            return -ENOMEM;
        }
        // spin_lock_init(&new_node->lock);
        // new_node->idle = true; // Initially, the channel is idle
        new_node->chan = chan;
        list_add_tail(&new_node->list, &chan_head);
        total_dma_channels++;
    }

    pr_info("%s: Total DMA channels allocated: %d\n", DEVICE_NAME, total_dma_channels);

    // convert the linked list to an array
    dma_channels = (struct dma_chan_node *) vmalloc(total_dma_channels * sizeof(struct dma_chan_node));
    if (!dma_channels) {
        pr_err("%s: Failed to allocate memory for DMA channel array\n", DEVICE_NAME);
        return -ENOMEM;
    }
   
    // init the dma channel array and delete the list
    list_for_each_entry_safe(curr_node, tmp, &chan_head, list) {
        list_del(&curr_node->list);
        dma_channels[i++].chan = curr_node->chan;
        // init_completion(&dma_channels[i].completion);

    }
#endif
    return 0;
}

static int fp_release(struct inode *inodep, struct file *filep) {
#if ENABLE_DMA
    int i = 0;
    for(i = 0; i < total_dma_channels; ++i){
        dma_release_channel(dma_channels[i].chan);
    }

    vfree(dma_channels);

    pr_info("%s: Released all %d DMA channels\n", DEVICE_NAME, total_dma_channels);
    total_dma_channels = 0;
#endif
    kfree(filep->private_data);
    return 0;
}

static int fp_mmap(struct file *filep, struct vm_area_struct *vma) {
    unsigned long req_size = vma->vm_end - vma->vm_start;
    unsigned long off   = 0;
    unsigned long uaddr = vma->vm_start;
    struct file_var *file_data = filep->private_data;
    if (!file_data || !file_data->target_buffer.vaddr_krnl) {
        pr_err("No cache buffer allocated for mmap\n");
        return -EINVAL;
    }

    pr_info("mmap: vaddr: %p paddr: %llx\n", file_data->target_buffer.vaddr_krnl, file_data->target_buffer.addr);

    if (req_size > file_data->target_buffer.size) {
        pr_err("Requested mmap size is larger than the allocated cache buffer size %lu > %llu\n",
               req_size, file_data->target_buffer.size);
        return -EINVAL;
    }
    // vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
    while (off < req_size) {
        struct page *p;
        int ret;
        void *kptr = (void *)(file_data->target_buffer.vaddr_krnl + off);

        p = virt_to_page(kptr);

        if (!p) return -EFAULT;

        ret = vm_insert_page(vma, (uaddr + off), p);
        if (ret && ret != -EBUSY) return ret;

        off += PAGE_SIZE;
    }

    // Map the buffer to the user space
    // if (remap_pfn_range(vma, vma->vm_start,
    //                     virt_to_phys(file_data->target_buffer.vaddr_krnl) >> PAGE_SHIFT,
    //                     req_size, vma->vm_page_prot)) {
    //     pr_err("Failed to remap user space to kernel buffer\n");
    //     return -EAGAIN;
    // }

    return 0;
}


long ioctl_allocate_cache_buffer(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    struct dma_buffer *cb = &file_data->target_buffer;
    pr_info("ioctl_allocate_cache_buffer");
    if (copy_from_user(cb, (struct dma_buffer *)arg, sizeof(*cb))) {
        return -EFAULT;
    }
#ifdef ALLOC_KMALLOC
    cb->vaddr_krnl = kmalloc(cb->size, GFP_KERNEL | GFP_DMA); // 4MB Max
    if (!cb->vaddr_krnl) {
        pr_err("Failed to allocate kernel memory\n");
        return -ENOMEM;
    }
#elif defined(ALLOC_GET_FREE_PAGES)
    cb->vaddr_krnl = (void *) __get_free_pages(GFP_KERNEL, get_order(cb->size)); // 4MB Max
    if (!cb->vaddr_krnl) {
        pr_err("Failed to allocate kernel memory\n");
        return -ENOMEM;
    }
#else
    dma_alloc_coherent(NULL, cb->size, &cb->addr, GFP_KERNEL); // error occur
#endif
    cb->addr = virt_to_phys(cb->vaddr_krnl);
    pr_info("Allocated cache buffer: vaddr_krnl=%p, addr: %llx size=%llu\n",
                cb->vaddr_krnl, cb->addr, cb->size);
    if (copy_to_user((struct dma_buffer *)arg, cb, sizeof(*cb))) {
                kfree(cb->vaddr_krnl);
                cb->vaddr_krnl = NULL;
                cb->addr = 0;
                cb->size = 0;
                return -EFAULT;
            }
    return 0;
}

long ioctl_set_cache_buffer(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    struct dma_buffer *cb = &file_data->target_buffer;
    if (copy_from_user(cb, (struct dma_buffer *)arg, sizeof(*cb))) {
        return -EFAULT;
    }
    return 0;
}


long ioctl_free_cache_buffer(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    struct dma_buffer *cb = &file_data->target_buffer;
    if (copy_from_user(cb, (struct dma_buffer *)arg, sizeof(*cb))) {
        pr_err("%s: Failed to copy cache buffer from user\n", DEVICE_NAME);
        return -EFAULT;
    }
    pr_info("Freeing kernel memory: vaddr_krnl=%p, addr: %llx size=%llu\n",
                cb->vaddr_krnl, cb->addr, cb->size);

#ifdef ALLOC_KMALLOC
    if (!cb->vaddr_krnl || 0 == cb->addr) {
        pr_err("No kernel memory allocated to free\n");
        return -EINVAL;
    }
    kfree(cb->vaddr_krnl);
#elif defined(ALLOC_GET_FREE_PAGES)
    if (!cb->vaddr_krnl) {
        pr_err("No kernel memory allocated to free\n");
        return -EINVAL;
    }
    free_pages((unsigned long)cb->vaddr_krnl, get_order(cb->size));
#else
    if (!cb->vaddr_krnl) {
        pr_err("No kernel memory allocated to free\n");
        return -EINVAL;
    }
    dma_free_coherent(NULL, cb->size, cb->vaddr_krnl, cb->addr);
#endif
    return 0;
}


static void update_cpl_to_hbm(struct dma_queue_pair * queue_pair, uint32_t identifier){
    struct agile_dma_cpl_t *cpls = queue_pair->cpls;
    uint32_t cq_tail;
    spin_lock(&queue_pair->cpl_lock);
    cq_tail = queue_pair->cq_tail;
    cpls[cq_tail].identifier = identifier;
    cpls[cq_tail].reserve = 0xaa;
    if(cpls[cq_tail].status == DMA_CPL_READY){
        pr_err("overwrite cpl to hbm\n");
    }
    cpls[cq_tail].status = DMA_CPL_READY;
    queue_pair->cq_tail = (cq_tail + 1) % queue_pair->depth;
    spin_unlock(&queue_pair->cpl_lock);
}


static void dma_callback(void *param)
{
    struct dma_callback_param *cb_param = param;
    // uint64_t finish_time = ktime_get_boottime_ns();
    // uint64_t issue_time = cb_param->issue_finish - cb_param->issue_start;
    // uint64_t dma_time = finish_time - cb_param->issue_finish;
    // uint64_t update_time;
    complete(cb_param->event);
    update_cpl_to_hbm(cb_param->queue_pair, cb_param->identifier);
    // update_time = ktime_get_boottime_ns() - finish_time;
    // pr_info("DMA transfer completed for src: %llx, dst: %llx, issue time: %lld, dma time: %lld, update time: %lld\n", cb_param->src_addr, cb_param->dst_addr, issue_time, dma_time, update_time);
}

long issue_dma_cmd(uint32_t engine_idx, uint64_t src_addr, uint64_t dst_addr, uint32_t size, struct dma_queue_pair *queue_pair, uint32_t cmd_pos, uint64_t issue_start){
    struct dma_chan_node *curr_node = &(dma_channels[engine_idx]);
    struct dma_async_tx_descriptor *tx = NULL;
    struct dma_callback_param *cb_param = &(queue_pair->cb_params[cmd_pos]);
    struct completion *event = &(queue_pair->events[cmd_pos]);
    enum dma_ctrl_flags flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
    dma_cookie_t cookie;
    // cb_param->issue_start = issue_start;

    // pr_info("Issuing DMA command on engine %d: src: %llx, dst: %llx, size: %u\n", engine_idx, src_addr, dst_addr, size);

    init_completion(event);
    // Prepare the DMA transaction
    tx = dmaengine_prep_dma_memcpy(curr_node->chan, dst_addr, src_addr, size, flags);
    if (!tx) {
        pr_err("dma_test: Failed to prepare DMA memcpy\n");
        return -EINVAL;
    }

    // Set callback parameters
    cb_param->src_addr = src_addr;
    cb_param->dst_addr = dst_addr;
    cb_param->size = size;
    cb_param->event = event;
    cb_param->queue_pair = queue_pair;
    cb_param->identifier = cmd_pos;

    tx->callback_param = cb_param;
    tx->callback = dma_callback;

    // Submit the transaction
    cookie = tx->tx_submit(tx);
    if (dma_submit_error(cookie)) {
        pr_err("dma_test: Failed to do tx_submit\n");
        return -EINVAL;
    }
    
    dma_async_issue_pending(curr_node->chan);
    return 0;
}

long ioctl_submit_dma_cmd(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    struct dma_queue_pair *queue_pair;
    struct agile_dma_cmd_t *cmds;
    uint32_t engine_idx = 0;
    uint32_t i = 0;
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    uint64_t issue_start; //  = ktime_get_boottime_ns();
    // struct dma_chan_node *curr_node;
    // struct dma_async_tx_descriptor *tx = NULL;
    // struct dma_callback_param *cb_param;
    // dma_cookie_t cookie;
    // struct completion *cpl;
    // enum dma_ctrl_flags flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
    // unsigned int dma_idx;
    struct dma_command dma_cmd;
    if (copy_from_user(&dma_cmd, (struct dma_command *)arg, sizeof(dma_cmd))) {
        pr_err("Failed to copy data from user\n");
        return -EFAULT;
    }

    for(i = 0; i < dma_cmd.count; ++i){
        queue_pair = &file_data->cpu_dma_queues[dma_cmd.queue_id];
        cmds = queue_pair->cmds;
        engine_idx = cmds[queue_pair->sq_head].dma_engine_id;

        if(cmds[queue_pair->sq_head].direction == DMA_CPU2GPU){
            src_addr = file_data->dram_offsets;
            dst_addr = file_data->hbm_offsets;
        } else if(cmds[queue_pair->sq_head].direction == DMA_GPU2CPU){
            dst_addr = file_data->dram_offsets;
            src_addr = file_data->hbm_offsets;
        } else {
            pr_err("Unknown direction\n");
            return -EFAULT;
        }

        src_addr += cmds[queue_pair->sq_head].src_offset;
        dst_addr += cmds[queue_pair->sq_head].dst_offset;

        if(issue_dma_cmd(engine_idx, src_addr, dst_addr, cmds[queue_pair->sq_head].size, queue_pair, queue_pair->sq_head, issue_start) != 0){
            pr_err("issue dma error\n");
            return -EFAULT;
        }
        queue_pair->sq_head = (queue_pair->sq_head + 1) % queue_pair->depth;
    }
    
    // curr_node = &(dma_channels[dma_idx]);

    // // Initialize completion
    // cpl = vmalloc(sizeof(struct completion));
    // if (!cpl) {
    //     pr_err("Failed to allocate memory for completion\n");
    //     return -ENOMEM;
    // }
    // init_completion(cpl);
    // cb_param = vmalloc(sizeof(struct dma_callback_param));
    // if (!cb_param) {
    //     pr_err("Failed to allocate memory for callback parameters\n");
    //     vfree(cpl);
    //     return -ENOMEM;
    // }
    
    // // Prepare the DMA transaction
    // tx = dmaengine_prep_dma_memcpy(curr_node->chan, dma_cmd.dst_addr, dma_cmd.src_addr, dma_cmd.size, flags);
    // if (!tx) {
    //     pr_err("dma_test: Failed to prepare DMA memcpy\n");
    //     return -EINVAL;
    // }

    // // Set callback parameters

    // cb_param->dma_src = dma_cmd.src_addr;
    // cb_param->dma_dst = dma_cmd.dst_addr;
    // cb_param->chan = curr_node->chan;
    // cb_param->size = dma_cmd.size;


    // tx->callback_param = cb_param;
    // tx->callback = dma_callback;

    // // Submit the transaction
    // cookie = tx->tx_submit(tx);
    // if (dma_submit_error(cookie)) {
    //     pr_err("dma_test: Failed to do tx_submit\n");
    //     return -EINVAL;
    // }

    // dma_async_issue_pending(curr_node->chan);

    // if(copy_to_user((struct dma_command *)arg, &dma_cmd, sizeof(dma_cmd))){
    //     pr_err("Failed to copy data to user\n");
    //     return -EFAULT;
    // }
    return 0;
}

long ioctl_set_cpu_dma_queue_num(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    uint32_t cpu_dma_queue_num;
    if (copy_from_user(&cpu_dma_queue_num, (uint32_t *)arg, sizeof(cpu_dma_queue_num))) {
        pr_err("Failed to copy data from user\n");
        return -EFAULT;
    }
    file_data->cpu_dma_queue_num = cpu_dma_queue_num;
    file_data->cpu_dma_queues = vmalloc(cpu_dma_queue_num * sizeof(struct dma_queue_pair));
    if(!file_data->cpu_dma_queues){
        pr_err("Failed to allocate memory for CPU DMA queues\n");
        return -ENOMEM;
    }
    memset(file_data->cpu_dma_queues, 0, cpu_dma_queue_num * sizeof(struct dma_queue_pair));
    pr_info("Set CPU DMA queue number to %u\n", cpu_dma_queue_num);
    return 0;
}

long ioctl_free_cpu_dma_queues(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    uint32_t cpu_dma_queue_num = file_data->cpu_dma_queue_num;
    uint32_t i;

    if(cpu_dma_queue_num == 0){
        pr_info("No CPU DMA queues to free\n");
        return 0;
    }

    if(!file_data->cpu_dma_queues){
        pr_info("No CPU DMA queues allocated\n");
        return 0;
    }

    for(i = 0; i < cpu_dma_queue_num; ++i){
        if(file_data->cpu_dma_queues[i].events){
            file_data->cpu_dma_queues[i].cmds = NULL;
            file_data->cpu_dma_queues[i].cpls = NULL;
            vfree(file_data->cpu_dma_queues[i].events);
            vfree(file_data->cpu_dma_queues[i].cb_params);
        }
    }
    
    vfree(file_data->cpu_dma_queues);
    file_data->cpu_dma_queues = NULL;
    file_data->cpu_dma_queue_num = 0;
    pr_info("Freed %u CPU DMA queues\n", cpu_dma_queue_num);
    return 0;
}

long ioctl_register_dma_queue_pairs(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    struct dma_queue_pair_data queue_data;
    uint32_t cpu_dma_queue_num = file_data->cpu_dma_queue_num;
    uint32_t g_idx = file_data->g_idx;
    if(cpu_dma_queue_num == 0){
        pr_err("CPU DMA queue number is not set\n");
        return -EINVAL;
    }

    if(g_idx >= cpu_dma_queue_num){
        pr_err("All CPU DMA queues are already registered\n");
        return -EINVAL;
    }

    if (copy_from_user(&queue_data, (struct dma_queue_pair_data *)arg, sizeof(queue_data))) {
        pr_err("Failed to copy data from user\n");
        return -EFAULT;
    }

    file_data->cpu_dma_queues[g_idx].id = g_idx;
    file_data->cpu_dma_queues[g_idx].sq_head = 0;
    file_data->cpu_dma_queues[g_idx].sq_tail = 0;
    file_data->cpu_dma_queues[g_idx].cq_head = 0;
    file_data->cpu_dma_queues[g_idx].cq_tail = 0;
    file_data->cpu_dma_queues[g_idx].depth = queue_data.queue_depth;
    file_data->cpu_dma_queues[g_idx].cmds = (struct agile_dma_cmd_t *) queue_data.cmds;
    file_data->cpu_dma_queues[g_idx].cpls = (struct agile_dma_cpl_t *) queue_data.cpls;
    file_data->cpu_dma_queues[g_idx].events = vmalloc(sizeof(struct completion) * queue_data.queue_depth);
    file_data->cpu_dma_queues[g_idx].cb_params = vmalloc(sizeof(struct dma_callback_param) * queue_data.queue_depth);
    spin_lock_init(&(file_data->cpu_dma_queues[g_idx].cpl_lock));
    pr_info("register dma queue %d ptr: %p val: %x\n", g_idx, queue_data.cmds, ((uint32_t *)queue_data.cmds)[0]);
    ((uint32_t *)queue_data.cmds)[0] = 0;

    // file_data->cpu_dma_queues[g_idx].cpls[0].identifier = g_idx + 10000;
    file_data->g_idx++;
    
    return 0;
}

long ioctl_set_base_addr_offsets(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    struct base_addr_offsets offsets;
    
    if(copy_from_user(&offsets, (struct base_addr_offsets*) arg, sizeof(offsets))){
        pr_err("ioctl_set_base_addr_offsets: Failed to copy data from usser\n");
    }
    file_data->dram_offsets = offsets.dram_offsets;
    file_data->hbm_offsets = offsets.hbm_offsets;
    pr_info("dram_offsets: 0x%llx hbm_offsets: 0x%llx\n", offsets.dram_offsets, offsets.hbm_offsets);
    return 0;
}

long ioctl_force_release_dma_channels(struct file *file, unsigned int cmd, unsigned long arg){
    int i = 0;
    for(i = 0; i < total_dma_channels; ++i){
        dma_release_channel(dma_channels[i].chan);
    }

    vfree(dma_channels);

    pr_info("%s: Force released all %d DMA channels\n", DEVICE_NAME, total_dma_channels);
    total_dma_channels = 0;
    return 0;
}

static long fp_ioctl(struct file *file, unsigned int cmd, unsigned long arg){
    // struct file_var *file_data = (struct file_var *) file->private_data;
    switch(cmd){
        case IOCTL_ALLOCATE_CACHE_BUFFER:
            return ioctl_allocate_cache_buffer(file, cmd, arg);
        case IOCTL_SET_CACHE_BUFFER:
            return ioctl_set_cache_buffer(file, cmd, arg);
        case IOCTL_FREE_CACHE_BUFFER:
            return ioctl_free_cache_buffer(file, cmd, arg);
        case IOCTL_SUBMIT_DMA_CMD:
            return ioctl_submit_dma_cmd(file, cmd, arg);
        case IOCTL_SET_CPU_DMA_QUEUE_NUM:
            return ioctl_set_cpu_dma_queue_num(file, cmd, arg);
        case IOCTL_FREE_CPU_DMA_QUEUES:
            return ioctl_free_cpu_dma_queues(file, cmd, arg);
        case IOCTL_REGISTER_DMA_QUEUE_PAIRS:
            return ioctl_register_dma_queue_pairs(file, cmd, arg);
        case IOCTL_SET_BASE_ADDR_OFFSETS:
            return ioctl_set_base_addr_offsets(file, cmd, arg);
        case IOCTL_FORCE_RELEASE_DMA_CHANNELS:
            return ioctl_force_release_dma_channels(file, cmd, arg);
        case IOCTL_GET_TOTAL_DMA_CHANNELS:
        {
            if (copy_to_user((uint32_t *)arg, &total_dma_channels, sizeof(total_dma_channels))) {
                return -EFAULT;
            }
            return 0;
        }
        default:
            pr_err("%s: Unsupported IOCTL command: %u\n", DEVICE_NAME, cmd);
            return -EINVAL;
    }
    return 0;
}

static struct file_operations fops = {
    .open = fp_open,
    .release = fp_release,
    .mmap = fp_mmap,
    .unlocked_ioctl = fp_ioctl, 
};

static int __init agile_kernel_init(void) {
    int ret;

    dma_cap_mask_t mask;
    struct dma_chan *chan;
    bool chan_available = true;

    // 1. Allocate a device number (major/minor)
    ret = alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        pr_err("Failed to alloc_chrdev_region\n");
        return ret;
    }

    // 2. Initialize and add cdev
    cdev_init(&kernel_cdev, &fops);
    kernel_cdev.owner = THIS_MODULE;
    ret = cdev_add(&kernel_cdev, dev_num, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev_num, 1);
        return ret;
    }

    // 3. Create a device class
    kernel_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(kernel_class)) {
        cdev_del(&kernel_cdev);
        unregister_chrdev_region(dev_num, 1);
        return PTR_ERR(kernel_class);
    }

    // 4. Create the device node in /dev
    if (IS_ERR(device_create(kernel_class, NULL, dev_num, NULL, DEVICE_NAME))) {
        class_destroy(kernel_class);
        cdev_del(&kernel_cdev);
        unregister_chrdev_region(dev_num, 1);
        return -1;
    }

    pr_info("%s: registered with major=%d minor=%d\n", DEVICE_NAME, MAJOR(dev_num), MINOR(dev_num));

    dma_cap_zero(mask);
    dma_cap_set(DMA_MEMCPY, mask);
    chan = dma_request_channel(mask, NULL, NULL);
    if (!chan) {
        chan_available = false;
        pr_err("%s: No DMA channel available\n", DEVICE_NAME);
        device_destroy(kernel_class, dev_num);
        class_destroy(kernel_class);
        cdev_del(&kernel_cdev);
        unregister_chrdev_region(dev_num, 1);
        pr_info("%s: unregistered\n", DEVICE_NAME);
        return -ENODEV;
    }
    pr_info("%s: DMA channel is available: %s\n", DEVICE_NAME, dev_name(chan->device->dev));
    dma_release_channel(chan);
    return 0;
}

static void __exit agile_kernel_exit(void) {
    device_destroy(kernel_class, dev_num);
    class_destroy(kernel_class);
    cdev_del(&kernel_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("%s: unregistered\n", DEVICE_NAME);
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zhuoping Yang");
MODULE_DESCRIPTION("AGILE Kernel Driver");
MODULE_VERSION("1.0");

module_init(agile_kernel_init);
module_exit(agile_kernel_exit);
