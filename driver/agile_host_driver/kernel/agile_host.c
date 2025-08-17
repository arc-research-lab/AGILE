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


#include "../common/agile_host_driver.h"


#define ALLOC_KMALLOC 1


#define DEVICE_NAME "AGILE-host"
#define CLASS_NAME  "AGILE Host Class"

static dev_t dev_num;

static dev_t dev_num;
static struct cdev  host_cdev;
static struct class *host_class;

static uint32_t total_dma_channels = 0;

struct dma_callback_param {
    struct completion *completion;
    bool *idle; // Indicates if the channel is idle
    unsigned int identifier; // command identifier
    unsigned int pid;       // process identifier
    dma_addr_t dma_src;
    dma_addr_t dma_dst;
    uint32_t size;
    struct dma_chan *chan;
    volatile uint32_t *cpl_ptr;
    struct eventfd_ctx *efd;
    int eventfd;
};

struct dma_chan_list {
    bool idle; // Indicates if the channel is idle

    struct dma_chan *chan;
    struct completion completion; // Completion for this channel
    struct dma_callback_param cb_param; // Callback parameters

    struct list_head list;
};

static LIST_HEAD(chan_head);

struct file_var {
    struct cache_buffer target_buffer;
};

static struct file_var *file_data;

static int fp_open(struct inode *inodep, struct file *filep) {

    dma_cap_mask_t mask;
    struct dma_chan *chan;
    bool chan_available = true;
    struct dma_chan_list *new_node;

    file_data = kmalloc(sizeof(struct file_var), GFP_KERNEL);
    if (!file_data) {
        return -ENOMEM;
    }
    filep->private_data = file_data;
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
    new_node->idle = true; // Initially, the channel is idle
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
        new_node->idle = true; // Initially, the channel is idle
        new_node->chan = chan;
        list_add_tail(&new_node->list, &chan_head);
        total_dma_channels++;
    }

    pr_info("%s: Total DMA channels allocated: %d\n", DEVICE_NAME, total_dma_channels);


    return 0;
}

static int fp_release(struct inode *inodep, struct file *filep) {

    struct dma_chan_list *curr_node, *tmp;

    list_for_each_entry_safe(curr_node, tmp, &chan_head, list) {
        list_del(&curr_node->list);
        dma_release_channel(curr_node->chan);
        vfree(curr_node);
    }
   
    // dma_idx = 0;

    pr_info("%s: Released all %d DMA channels\n", DEVICE_NAME, total_dma_channels);
    total_dma_channels = 0;

    kfree(file_data);
    return 0;
}

static int fp_mmap(struct file *filep, struct vm_area_struct *vma) {
    unsigned long req_size = vma->vm_end - vma->vm_start;

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
    struct eventfd_ctx *efd = cb_param->efd;
    // struct task_struct *task;
    // unsigned long flags;

    complete(cb_param->completion);
    // For demonstration, we will just simulate a signal to the process
    // rcu_read_lock();
    // if(*(cb_param->cpl) != 0){
    //     pr_err("DMA transfer failed for PID %d with identifier %u cpl not initialized: %d\n", cb_param->pid, cb_param->identifier, *(cb_param->cpl));
    // }else{
    //     // *(cb_param->cpl) = 1;
    // }
    // *(cb_param->idle) = true; // Mark the channel as idle again
    // spin_unlock_irqrestore(cb_param->lock, flags);
    // rcu_read_unlock();
    
    *(cb_param->cpl_ptr) = 1;
    
    if (efd != -EBADF && efd != NULL) {
        pr_info("Got eventfd context: %p fd: %d\n", efd, cb_param->eventfd);
        eventfd_signal(efd, 1);
        // eventfd_signal(efd, 1);
    }else{
        pr_err("Failed to get eventfd context\n");
    }

//dma_unmap_single(cb_param->chan->device->dev, cb_param->dma_src, cb_param->size, DMA_TO_DEVICE);
//    dma_unmap_single(cb_param->chan->device->dev, cb_param->dma_dst, cb_param->size, DMA_FROM_DEVICE);
    pr_info("DMA transfer completed for PID %d with identifier %u\n", cb_param->pid, cb_param->identifier);
}

static long fp_ioctl(struct file *file, unsigned int cmd, unsigned long arg){
    struct cache_buffer *cb = &file_data->target_buffer;
    switch(cmd){
        case IOCTL_ALLOCATE_CACHE_BUFFER:
            if (copy_from_user(cb, (struct cache_buffer *)arg, sizeof(*cb))) {
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
            if (copy_to_user((struct cache_buffer *)arg, cb, sizeof(*cb))) {
                // dma_free_coherent(NULL, cb->size, cb->vaddr_krnl, cb->addr);
                kfree(cb->vaddr_krnl);
                cb->vaddr_krnl = NULL;
                cb->addr = 0;
                cb->size = 0;
                return -EFAULT;
            }
            break;
        case IOCTL_SET_CACHE_BUFFER:
            if (copy_from_user(&cb, (struct cache_buffer *)arg, sizeof(cb))) {
                return -EFAULT;
            }
            // Here you would typically set the cache buffer in the driver
            break;
        case IOCTL_FREE_CACHE_BUFFER:
            if (copy_from_user(cb, (struct cache_buffer *)arg, sizeof(*cb))) {
                pr_err("%s: Failed to copy cache buffer from user\n", DEVICE_NAME);
                return -EFAULT;
            }
#ifdef ALLOC_KMALLOC
            if (!cb->vaddr_krnl) {
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
        case IOCTL_SUBMIT_DMA_CMD:
        {
            struct dma_command dma_cmd;
            struct dma_chan_list *curr_node;
            struct dma_async_tx_descriptor *tx = NULL;
            dma_addr_t dma_src, dma_dst;
            dma_cookie_t cookie;
            enum dma_ctrl_flags flags = DMA_CTRL_ACK | DMA_PREP_INTERRUPT;
            bool find_dma_channel = false;

            pr_info("dma_test: Received DMA command from user space\n");

            // Copy DMA command from user space
            if (copy_from_user(&dma_cmd, (struct dma_command *)arg, sizeof(dma_cmd))) {
                pr_err("Failed to copy data from user\n");
                return -EFAULT;
            }


            // Find an available DMA channel
            FIND_DMA_CHANNEL:
            list_for_each_entry(curr_node, &chan_head, list) {
                pr_info("Checking DMA Channel: %s\n", dev_name(curr_node->chan->device->dev));
                // spin_lock_irqsave(&curr_node->lock, flags);
                if(curr_node->idle){
                    curr_node->idle = false; // Mark the channel as busy
                    // spin_unlock_irqrestore(&curr_node->lock, flags);
                    find_dma_channel = true;
                    break;
                }
                // spin_unlock_irqrestore(&curr_node->lock, flags);
            }
            if (!find_dma_channel) {
                pr_info("No idle DMA channel found, retrying...\n");
                goto FIND_DMA_CHANNEL; // If no idle channel found, try again
            }

            // Map the contigous buffers to DMA addresses
            dma_src = dma_map_single(curr_node->chan->device->dev, dma_cmd.src_vaddr_krnl, dma_cmd.size, DMA_TO_DEVICE);
            dma_dst = dma_map_single(curr_node->chan->device->dev, dma_cmd.dst_vaddr_krnl, dma_cmd.size, DMA_FROM_DEVICE);
            if (dma_mapping_error(curr_node->chan->device->dev, dma_src) ||
                dma_mapping_error(curr_node->chan->device->dev, dma_dst)) {
                pr_err("dma_test: DMA mapping failed\n");
                return -EINVAL;
            }

            pr_info("dma_test: src_buf mapped to %llx physical addr: %llx, dst_buf mapped to %llx physical addr: %llx\n", dma_src, virt_to_phys(dma_cmd.src_vaddr_krnl), dma_dst, virt_to_phys(dma_cmd.dst_vaddr_krnl));

            // Prepare the DMA transaction
            tx = dmaengine_prep_dma_memcpy(curr_node->chan, dma_dst, dma_src, dma_cmd.size, flags);
            if (!tx) {
                pr_err("dma_test: Failed to prepare DMA memcpy\n");
                dma_unmap_single(curr_node->chan->device->dev, dma_src, dma_cmd.size, DMA_TO_DEVICE);
                dma_unmap_single(curr_node->chan->device->dev, dma_dst, dma_cmd.size, DMA_FROM_DEVICE);
                return -EINVAL;
            }

            // Initialize completion
            init_completion(&curr_node->completion);

            // Set callback parameters
            curr_node->cb_param.completion = &curr_node->completion;
            curr_node->cb_param.identifier = dma_cmd.identifier;
            curr_node->cb_param.pid = dma_cmd.pid;
            curr_node->cb_param.idle = &curr_node->idle; // Pass the idle flag
            curr_node->cb_param.dma_src = dma_src;
            curr_node->cb_param.dma_dst = dma_dst;
            curr_node->cb_param.chan = curr_node->chan;
            curr_node->cb_param.cpl_ptr = dma_cmd.cpl_ptr;
            curr_node->cb_param.eventfd = dma_cmd.eventfd;
            curr_node->cb_param.efd = eventfd_ctx_fdget(dma_cmd.eventfd);

            tx->callback_param = &curr_node->cb_param;
            tx->callback = dma_callback;
            
            // Submit the transaction
            cookie = tx->tx_submit(tx);
            if (dma_submit_error(cookie)) {
                pr_err("dma_test: Failed to do tx_submit\n");
                dma_unmap_single(curr_node->chan->device->dev, dma_src, dma_cmd.size, DMA_TO_DEVICE);
                dma_unmap_single(curr_node->chan->device->dev, dma_dst, dma_cmd.size, DMA_FROM_DEVICE);
                return -EINVAL;
            }

            // Trigger the DMA transaction
            dma_async_issue_pending(curr_node->chan);

            pr_info("Submitting DMA command: src_addr=%llx, dst_addr=%llx, size=%u, direction=%u, identifier=%u, pid=%u\n",
                dma_cmd.src_addr, dma_cmd.dst_addr, dma_cmd.size,
                dma_cmd.direction, dma_cmd.identifier, dma_cmd.pid);


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

static int __init agile_host_init(void) {
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
    cdev_init(&host_cdev, &fops);
    host_cdev.owner = THIS_MODULE;
    ret = cdev_add(&host_cdev, dev_num, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev_num, 1);
        return ret;
    }

    // 3. Create a device class
    host_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(host_class)) {
        cdev_del(&host_cdev);
        unregister_chrdev_region(dev_num, 1);
        return PTR_ERR(host_class);
    }

    // 4. Create the device node in /dev
    if (IS_ERR(device_create(host_class, NULL, dev_num, NULL, DEVICE_NAME))) {
        class_destroy(host_class);
        cdev_del(&host_cdev);
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
        device_destroy(host_class, dev_num);
        class_destroy(host_class);
        cdev_del(&host_cdev);
        unregister_chrdev_region(dev_num, 1);
        pr_info("%s: unregistered\n", DEVICE_NAME);
        return -ENODEV;
    }
    pr_info("%s: DMA channel is available: %s\n", DEVICE_NAME, dev_name(chan->device->dev));
    dma_release_channel(chan);
    return 0;
}

static void __exit agile_host_exit(void) {
    device_destroy(host_class, dev_num);
    class_destroy(host_class);
    cdev_del(&host_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("%s: unregistered\n", DEVICE_NAME);
}

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zhuoping Yang");
MODULE_DESCRIPTION("Kernel module to allocate contiguous CPU memory and mmap to user");
MODULE_VERSION("1.0");

module_init(agile_host_init);
module_exit(agile_host_exit);
