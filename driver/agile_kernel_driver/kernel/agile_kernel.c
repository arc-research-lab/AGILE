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

#include "../common/agile_kernel_driver.h"


#define ALLOC_KMALLOC 1
#define DEVICE_NAME "AGILE-kernel"
#define CLASS_NAME  "AGILE Kernel Class"

#define ENABLE_DMA 1


struct dma_callback_param {
    struct completion *completion;
    uint32_t coroutine_idx; // command identifier
    uint32_t cpl_queue_idx;
    dma_addr_t dma_src;
    dma_addr_t dma_dst;
    uint32_t size;
    struct dma_chan *chan;
    struct dma_cpl_queue *cpl_queue;
    // struct eventfd_ctx *efd;
};

struct dma_chan_list {
    struct dma_chan *chan;
    struct list_head list;
};

struct dma_chan_node {
    struct dma_chan *chan;
};

struct dma_cpl_queue {
    union dma_cpl_queue_data data;
    uint32_t head;
    uint32_t tail;
    uint32_t depth;
    struct eventfd_ctx *efd_ctx;
    spinlock_t lock;
};

struct dma_cpl_queues_info {
    struct dma_cpl_queue *queues;
    int num_queues;
};

struct file_var {
    struct dma_buffer target_buffer;
    struct dma_cpl_queues_info cpl_queues_info;
};


static LIST_HEAD(chan_head);

static dev_t dev_num;

static dev_t dev_num;
static struct cdev  kernel_cdev;
static struct class *kernel_class;

static uint32_t total_dma_channels = 0;
// static atomic64_t g_dma_idx = ATOMIC64_INIT(0);
static struct dma_chan_node * dma_channels;


static int fp_open(struct inode *inodep, struct file *filep) {
#if ENABLE_DMA
    dma_cap_mask_t mask;
    struct dma_chan *chan;
    bool chan_available = true;
    struct dma_chan_list *new_node;
    struct dma_chan_list *curr_node, *tmp;
    int i = 0;

    filep->private_data = kzalloc(sizeof(struct file_var), GFP_KERNEL);
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
    struct file_var *file_data = filep->private_data;
    if (!file_data || !file_data->target_buffer.vaddr_krnl) {
        pr_err("No cache buffer allocated for mmap\n");
        return -EINVAL;
    }

    if (req_size > file_data->target_buffer.size) {
        pr_err("Requested mmap size is larger than the allocated cache buffer size %lu > %llu\n",
               req_size, file_data->target_buffer.size);
        return -EINVAL;
    }

    // Map the buffer to the user space
    if (remap_pfn_range(vma, vma->vm_start,
                        virt_to_phys(file_data->target_buffer.vaddr_krnl) >> PAGE_SHIFT,
                        req_size, vma->vm_page_prot)) {
        pr_err("Failed to remap user space to kernel buffer\n");
        return -EAGAIN;
    }

    return 0;
}

static void dma_callback(void *param)
{
    struct dma_callback_param *cb_param = param;
    // uint32_t queue_idx = cb_param->cpl_queue_idx;
    struct dma_cpl_queue * cpl_queue = cb_param->cpl_queue;
    struct eventfd_ctx *efd = cpl_queue->efd_ctx;

    complete(cb_param->completion);

    // add completion to the completion queue
    // spin_lock(&cpl_queue->lock);
    if(cpl_queue->data.entry[cpl_queue->tail].coroutine_idx != 0){
        pr_err("Completion queue is full\n");
    }
    cpl_queue->data.entry[cpl_queue->tail].coroutine_idx = cb_param->coroutine_idx;
    cpl_queue->tail = (cpl_queue->tail + 1) % cpl_queue->depth;
    // spin_unlock(&cpl_queue->lock);

    if (!IS_ERR(efd) && efd != NULL) {
        eventfd_signal(efd, 1);
    } else {
        pr_err("Failed to get eventfd context %p\n", efd);
    }

    // dma_unmap_single(cb_param->chan->device->dev, cb_param->dma_src, cb_param->size, DMA_TO_DEVICE);
    // dma_unmap_single(cb_param->chan->device->dev, cb_param->dma_dst, cb_param->size, DMA_FROM_DEVICE);
    
    pr_info("DMA transfer completed for coroutine_idx %u\n", cb_param->coroutine_idx);
    vfree(cb_param->completion);
    vfree(cb_param);
}

int ioctl_allocate_cache_buffer(struct dma_buffer *cb){
#ifdef ALLOC_KMALLOC
        cb->vaddr_krnl = kzalloc(cb->size, GFP_KERNEL | GFP_DMA); // 4MB Max
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

        return 0;
}

int ioctl_free_cache_buffer(struct dma_buffer *cb){
#ifdef ALLOC_KMALLOC
            if (!cb->vaddr_krnl || 0 == cb->addr) {
                pr_err("No kernel memory allocated to free\n");
                return -EINVAL;
            }
            pr_info("Freeing kernel memory: vaddr_krnl=%p, addr: %llx size=%llu\n",
                     cb->vaddr_krnl, cb->addr, cb->size);
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


int ioctl_submit_dma_cmd(struct file_var *file_data, struct dma_command *dma_cmd){
    struct dma_chan_node *curr_node;
    struct dma_async_tx_descriptor *tx = NULL;
    struct dma_callback_param *cb_param;
    // dma_addr_t dma_src, dma_dst;
    dma_cookie_t cookie;
    struct completion *cpl;
    enum dma_ctrl_flags flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
    // struct dma_cpl_queue * cpl_queue = &(file_data->cpl_queues_info.queues[dma_cmd->cpl_queue_idx]);

    unsigned int dma_idx;
    dma_idx = dma_cmd->dma_channel_idx;

    pr_info("Submitting DMA command on channel %u; src_addr: 0x%llx, dst_addr: 0x%llx\n", dma_idx, dma_cmd->src_addr, dma_cmd->dst_addr);

    curr_node = &(dma_channels[dma_idx]);

    // Initialize completion
    cpl = vmalloc(sizeof(struct completion));
    if (!cpl) {
        pr_err("Failed to allocate memory for completion\n");
        return -ENOMEM;
    }
    init_completion(cpl);
    cb_param = vmalloc(sizeof(struct dma_callback_param));
    if (!cb_param) {
        pr_err("Failed to allocate memory for callback parameters\n");
        vfree(cpl);
        return -ENOMEM;
    }
    

    // Map the contigous buffers to DMA addresses
    // dma_src = dma_map_single(curr_node->chan->device->dev, dma_cmd->src_vaddr_krnl, dma_cmd->size, DMA_TO_DEVICE);
    // dma_dst = dma_map_single(curr_node->chan->device->dev, dma_cmd->dst_vaddr_krnl, dma_cmd->size, DMA_FROM_DEVICE);
    // if (dma_mapping_error(curr_node->chan->device->dev, dma_src) ||
    //     dma_mapping_error(curr_node->chan->device->dev, dma_dst)) {
    //     pr_err("dma_test: DMA mapping failed\n");
    //     return -EINVAL;
    // }

    // Prepare the DMA transaction
    tx = dmaengine_prep_dma_memcpy(curr_node->chan, dma_cmd->dst_addr, dma_cmd->src_addr, dma_cmd->size, flags);
    if (!tx) {
        pr_err("dma_test: Failed to prepare DMA memcpy\n");
        // dma_unmap_single(curr_node->chan->device->dev, dma_src, dma_cmd->size, DMA_TO_DEVICE);
        // dma_unmap_single(curr_node->chan->device->dev, dma_dst, dma_cmd->size, DMA_FROM_DEVICE);
        return -EINVAL;
    }

    // Set callback parameters
    cb_param->completion = cpl;
    cb_param->coroutine_idx = dma_cmd->coroutine_idx;
    cb_param->cpl_queue = &file_data->cpl_queues_info.queues[dma_cmd->cpl_queue_idx];

    cb_param->dma_src = dma_cmd->src_addr;
    cb_param->dma_dst = dma_cmd->dst_addr;
    cb_param->chan = curr_node->chan;
    cb_param->size = dma_cmd->size;


    tx->callback_param = cb_param;
    tx->callback = dma_callback;

    // Submit the transaction
    cookie = tx->tx_submit(tx);
    if (dma_submit_error(cookie)) {
        pr_err("dma_test: Failed to do tx_submit\n");
        // dma_unmap_single(curr_node->chan->device->dev, dma_src, dma_cmd->size, DMA_TO_DEVICE);
        // dma_unmap_single(curr_node->chan->device->dev, dma_dst, dma_cmd->size, DMA_FROM_DEVICE);
        return -EINVAL;
    }

    dma_async_issue_pending(curr_node->chan);
    return 0;
}


int ioctl_allocate_cpl_queue_array(struct file_var *file_data, uint32_t num_queues) {
    file_data->cpl_queues_info.queues = vmalloc(num_queues * sizeof(struct dma_cpl_queue *));
    if (!file_data->cpl_queues_info.queues) {
        return -ENOMEM;
    }
    if(num_queues > 128){
        return -ENOMEM;
    }
    pr_info("Allocated %u completion queues\n", num_queues);
    file_data->cpl_queues_info.num_queues = num_queues;
    return 0;
}

int ioctl_set_cpl_queue(struct file_var *file_data, struct cpl_queue_config *cfg) {
    
    // Initialize the completion queue with the given configuration
    struct dma_cpl_queue * cpl_queue = &(file_data->cpl_queues_info.queues[cfg->idx]);
    cpl_queue->data.ptr = cfg->vaddr_krnl;
    cpl_queue->depth = cfg->depth;
    cpl_queue->head = 0;
    cpl_queue->tail = 0;
    cpl_queue->efd_ctx = eventfd_ctx_fdget(cfg->efd);
    spin_lock_init(&cpl_queue->lock);
    pr_info("efd: %d ctx: %p\n", cfg->efd, cpl_queue->efd_ctx);
    return 0;
}


int ioctl_del_cpl_queue(struct file_var *file_data, struct cpl_queue_config *cfg){
    file_data->cpl_queues_info.queues[cfg->idx] = (struct dma_cpl_queue){0};
    return 0;
}


int ioctl_test_dma_emu(struct file_var *file_data, struct dma_command *dma_cmd){
    struct dma_cpl_queue * cpl_queue = &(file_data->cpl_queues_info.queues[dma_cmd->cpl_queue_idx]);
    // uint32_t head = cpl_queue->head;
    uint32_t dma_channel_idx = dma_cmd->dma_channel_idx;
    pr_info("test dma emulation(chan-%d): src_addr=%llx, dst_addr=%llx, size=%u, coroutine_idx=%llu\n",
        dma_channel_idx, dma_cmd->src_addr, dma_cmd->dst_addr, dma_cmd->size,
        dma_cmd->coroutine_idx);

    cpl_queue->data.entry[cpl_queue->tail].coroutine_idx = dma_cmd->coroutine_idx;
    cpl_queue->tail = (cpl_queue->tail + 1) % cpl_queue->depth;

    // notify the completion
    // eventfd_signal(cpl_queue->efd_ctx, 1);
    if (!IS_ERR(cpl_queue->efd_ctx) && cpl_queue->efd_ctx != NULL) {
        pr_info("Got eventfd context: %p \n", cpl_queue->efd_ctx);
        eventfd_signal(cpl_queue->efd_ctx, 1);
    }else{
        pr_err("Failed to get eventfd context: %p \n", cpl_queue->efd_ctx);
    }

    return 0;
}

int ioctl_update_cpl_queue_head(struct file_var *file_data, struct cpl_queue_upd *upd) {
    struct dma_cpl_queue *cpl_queue = &(file_data->cpl_queues_info.queues[upd->idx]);
    cpl_queue->head = upd->head;
    return 0;
}

static int test_eventfd;
static struct eventfd_ctx *test_eventfd_ctx;
static long fp_ioctl(struct file *file, unsigned int cmd, unsigned long arg){
    struct file_var *file_data = (struct file_var *) file->private_data;
    switch(cmd){

        case IOCTL_ALLOCATE_CACHE_BUFFER:
        {
            struct dma_buffer *cb = &file_data->target_buffer;
            if (copy_from_user(cb, (struct dma_buffer *)arg, sizeof(*cb))) {
                return -EFAULT;
            }

            if (ioctl_allocate_cache_buffer(cb) != 0) {
                return -ENOMEM;
            }

            if (copy_to_user((struct dma_buffer *)arg, cb, sizeof(*cb))) {
                kfree(cb->vaddr_krnl);
                cb->vaddr_krnl = NULL;
                cb->addr = 0;
                cb->size = 0;
                return -EFAULT;
            }
            return 0;
        }

        case IOCTL_SET_CACHE_BUFFER:
        {
            struct dma_buffer *cb = &file_data->target_buffer;
            if (copy_from_user(cb, (struct dma_buffer *)arg, sizeof(*cb))) {
                return -EFAULT;
            }
            return 0;
        }

        case IOCTL_FREE_CACHE_BUFFER:
        {
            struct dma_buffer *cb = &file_data->target_buffer;
            if (copy_from_user(cb, (struct dma_buffer *)arg, sizeof(*cb))) {
                pr_err("%s: Failed to copy cache buffer from user\n", DEVICE_NAME);
                return -EFAULT;
            }
            return ioctl_free_cache_buffer(cb);
        }
            
        case IOCTL_SUBMIT_DMA_CMD:
        {
            struct dma_command dma_cmd;
            if (copy_from_user(&dma_cmd, (struct dma_command *)arg, sizeof(dma_cmd))) {
                pr_err("Failed to copy data from user\n");
                return -EFAULT;
            }
            return ioctl_submit_dma_cmd(file_data, &dma_cmd);
        }

        case IOCTL_ALLOC_CPL_QUEUE_ARRAY:
        {
            uint32_t num_queues;
            if (copy_from_user(&num_queues, (void __user *)arg, sizeof(num_queues))) {
                return -EFAULT;
            }
            return ioctl_allocate_cpl_queue_array(file_data, num_queues);
        }

        case IOCTL_FREE_CPL_QUEUE_ARRAY:
        {
            vfree(file_data->cpl_queues_info.queues);
            file_data->cpl_queues_info.queues = NULL;
            return 0;
        }
        
        case IOCTL_SET_CPL_QUEUE:
        {
            struct cpl_queue_config cfg;
            if (copy_from_user(&cfg, (struct cpl_queue_config *)arg, sizeof(cfg))) {
                return -EFAULT;
            }
            return ioctl_set_cpl_queue(file_data, &cfg);
        }

        case IOCTL_DEL_CPL_QUEUE:
        {
            struct cpl_queue_config cfg;
            if (copy_from_user(&cfg, (struct cpl_queue_config *)arg, sizeof(cfg))) {
                return -EFAULT;
            }
            return ioctl_del_cpl_queue(file_data, &cfg);
        }

        case IOCTL_GET_TOTAL_DMA_CHANNELS:
        {
            if (copy_to_user((uint32_t *)arg, &total_dma_channels, sizeof(total_dma_channels))) {
                return -EFAULT;
            }
            return 0;
        }

        case IOCTL_UPDATE_CPL_QUEUE_HEAD:
        {
            struct cpl_queue_upd upd;
            if (copy_from_user(&upd, (struct cpl_queue_upd *)arg, sizeof(upd))) {
                return -EFAULT;
            }
            return ioctl_update_cpl_queue_head(file_data, &upd);
        }

        case IOCTL_TEST_REGISTER_EVENTFD:
        {
            if(copy_from_user(&test_eventfd, (int *)arg, sizeof(test_eventfd))) {
                return -EFAULT;
            }
            test_eventfd_ctx = eventfd_ctx_fdget(test_eventfd);
            pr_info("eventfd: %d ctx: %p\n", test_eventfd, test_eventfd_ctx);
            return 0;
        }

        case IOCTL_TEST_SEND_SIG:
        {
            if (test_eventfd_ctx) {
                pr_info("Sending signal to eventfd %d\n", test_eventfd);
                eventfd_signal(test_eventfd_ctx, 1);
            }
            return 0;
        }

        case IOCTL_TEST_DMA_EMU:
        {
            struct dma_command dma_cmd;
            if (copy_from_user(&dma_cmd, (struct dma_command *)arg, sizeof(dma_cmd))) {
                pr_err("Failed to copy data from user\n");
                return -EFAULT;
            }
            return ioctl_test_dma_emu(file_data, &dma_cmd);
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
