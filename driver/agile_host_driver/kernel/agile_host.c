#include <linux/module.h>
#include <linux/slab.h>         
#include <linux/fs.h>           
#include <linux/cdev.h>
#include <linux/device.h>
#include <asm/io.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>

#include "../common/agile_host_driver.h"

#define DEVICE_NAME "AGILE-host"
#define CLASS_NAME  "AGILE Host Class"

static dev_t dev_num;

static dev_t dev_num;
static struct cdev  host_cdev;
static struct class *host_class;

struct file_var {
    struct cache_buffer target_buffer;
};

static struct file_var *file_data;

static int fp_open(struct inode *inodep, struct file *filep) {
    file_data = kmalloc(sizeof(struct file_var), GFP_KERNEL);
    if (!file_data) {
        return -ENOMEM;
    }
    filep->private_data = file_data;
    return 0;
}

static int fp_release(struct inode *inodep, struct file *filep) {
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

static long fp_ioctl(struct file *file, unsigned int cmd, unsigned long arg){
    struct cache_buffer *cb = &file_data->target_buffer;
    switch(cmd){
        case IOCTL_ALLOCATE_CACHE_BUFFER:
            if (copy_from_user(cb, (struct cache_buffer *)arg, sizeof(*cb))) {
                return -EFAULT;
            }
            // cb->vaddr_krnl = kmalloc(cb->size, GFP_KERNEL); // 4MB Max
            cb->vaddr_krnl = (void *) __get_free_pages(GFP_KERNEL, get_order(cb->size)); // 4MB Max
            // dma_alloc_coherent(NULL, cb->size, &cb->addr, GFP_KERNEL); // error occur
            if (!cb->vaddr_krnl) {
                pr_err("Failed to allocate kernel memory\n");
                return -ENOMEM;
            }
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
                return -EFAULT;
            }
            kfree(cb->vaddr_krnl);
            break;
        default:            
            pr_err("Unsupported IOCTL command: %u\n", cmd);
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