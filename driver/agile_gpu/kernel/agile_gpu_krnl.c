#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>
#include <linux/ioctl.h>
#include <linux/io.h>
#include <linux/slab.h>

#include "../common/agile_gpu_krnl.h"

// #ifdef _CUDA
#include <nv-p2p.h>

#ifndef NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API
#error "Current implementation requires NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API. Please use a driver/NVIDIA peer-to-peer SDK that provides it."
#endif

#define DEVICE_NAME "AGILE-gpu"

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zhuoping Yang");
MODULE_DESCRIPTION("Pin GPU Memory and map it to user space");
MODULE_VERSION("1.0");

struct file_vars{
    struct pin_buffer_params * curr_buf;
};

static int fp_mmap(struct file *filp, struct vm_area_struct *vma) {
    struct file_vars *vars = filp->private_data;
    struct pin_buffer_params * params = vars->curr_buf;
    unsigned long pfn;
    size_t size = vma->vm_end - vma->vm_start;
    if(!params){
        pr_err("No pinned buffer info found for this file\n");
        return -EINVAL;
    }
    if(size > params->size){
        pr_err("Requested mmap size %zu is larger than pinned buffer size %llu\n", size, params->size);
        return -EINVAL;
    }
    pfn = params->phy_addr >> PAGE_SHIFT;
    pr_info("mmaping phy mem addr=0x%llx size=%zu at user virt addr=0x%lx\n", 
             params->phy_addr, size, vma->vm_start);
    if (io_remap_pfn_range(vma, vma->vm_start, pfn, size, vma->vm_page_prot)) {
        pr_err("error in gdrdrv_io_remap_pfn_range()\n");
        return -EAGAIN;
    }
    return 0;
}

static void free_pages_callback(void *data){
    // mapping_info *minfo = (mapping_info *)data;
    pr_info("GPU pages freed callback invoked\n");
}

long ioctl_pin_gpu_buffer(struct file *file, unsigned int cmd, unsigned long arg) {
    
    struct nvidia_p2p_page_table *page_table;
    struct file_vars *vars = file->private_data;
    struct pin_buffer_params * params = vars->curr_buf;
    int i = 0;
    if (copy_from_user(params, (struct pin_buffer_params *)arg, sizeof(*params))) {
        return -EACCES;
    }

    // pin GPU buffer
    if (nvidia_p2p_get_pages(params->p2p_token, params->va_space, params->vaddr, params->size, &page_table, free_pages_callback, NULL) != 0) {
        pr_err("nvidia_p2p_get_pages failed\n");
        return -EACCES;
    }

    pr_info("nvidia_p2p_get_pages succeeded: entries=%d page_size=%d\n", page_table->entries, page_table->page_size);

    // check if the buffer is contiguous
    for(i = 1; i < page_table->entries; i++){
        if(page_table->pages[i]->physical_address != page_table->pages[i-1]->physical_address + 65536l && page_table->page_size == NVIDIA_P2P_PAGE_SIZE_64KB){
            pr_err("The pinned GPU buffer is not physically contiguous, cannot mmap to user space: addr-%d: 0x%llx addr-%d: 0x%llx\n", i - 1, page_table->pages[i - 1]->physical_address, i, page_table->pages[i]->physical_address);
            nvidia_p2p_put_pages(params->p2p_token, params->va_space, params->vaddr, page_table);
            return -EINVAL;
        }
    }

    params->phy_addr = page_table->pages[0]->physical_address;
    params->krnl_ptr = ioremap_wc(params->phy_addr, params->size);
    if(!params->krnl_ptr){
        pr_err("ioremap_wc failed\n");
        nvidia_p2p_put_pages(params->p2p_token, params->va_space, params->vaddr, page_table);
        return -EACCES;
    }
    ((uint32_t *) params->krnl_ptr)[0] = 10086; // test write
    pr_info("The pinned GPU buffer is physically contiguous, phy_addr=0x%llx\n", params->phy_addr);
    params->page_table = page_table;
    vars->curr_buf = params;
    if (copy_to_user((struct pin_buffer_params *)arg, params, sizeof(*params))) {
        return -EACCES;
    }
    return 0;
}

long ioctl_unpin_gpu_buffer(struct file *file, unsigned int cmd, unsigned long arg) {
    struct pin_buffer_params params;
    if (copy_from_user(&params, (struct pin_buffer_params *)arg, sizeof(params))) {
        return -EACCES;
    }
    if(params.page_table == NULL){
        pr_err("No pinned buffer info found for this file\n");
        return -EINVAL;
    }
    if(params.krnl_ptr){
        iounmap(params.krnl_ptr);
        params.krnl_ptr = NULL;
    }
    // unpin GPU buffer
    if(nvidia_p2p_put_pages(params.p2p_token, params.va_space, params.vaddr, params.page_table) != 0) {
        return -EACCES;
    }
    pr_info("Unpinned GPU buffer: vaddr=0x%llx size=%llu phy_addr=0x%llx\n", params.vaddr, params.size, params.phy_addr);
    return 0;
}



static long fp_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    switch(cmd){
        case IOCTL_PIN_GPU_BUFFER:
            return ioctl_pin_gpu_buffer(file, cmd, arg);
        case IOCTL_UNPIN_GPU_BUFFER:
            return ioctl_unpin_gpu_buffer(file, cmd, arg);
        default:
            return -EINVAL;
    }
    return 0;
}

static int fp_open(struct inode *inodep, struct file *filep) {
    
    struct file_vars *vars = kmalloc(sizeof(struct file_vars), GFP_KERNEL);
    if (!vars) {
        return -ENOMEM;
    }
    vars->curr_buf = kmalloc(sizeof(struct pin_buffer_params), GFP_KERNEL);
    if (!vars->curr_buf) {
        kfree(vars);
        return -ENOMEM;
    }
    memset(vars->curr_buf, 0, sizeof(struct pin_buffer_params));
    pr_info("%s Opened\n", DEVICE_NAME);
    filep->private_data = vars;
    return 0;
}

static int fp_release(struct inode *inodep, struct file *filep) {
    
    struct file_vars *vars = filep->private_data;
    pr_info("%s Closed\n", DEVICE_NAME);
    kfree(vars->curr_buf);
    kfree(vars);
    return 0;
}

static const struct file_operations fp_fops = {
    .owner = THIS_MODULE,
    .mmap = fp_mmap,
    .open = fp_open,
    .release = fp_release,
    .unlocked_ioctl = fp_ioctl,
};

static struct miscdevice fp_dev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = DEVICE_NAME,
    .fops = &fp_fops,
};

static int __init AGILE_GPU_init(void)
{
    pr_info("%s Init\n", DEVICE_NAME);
    return misc_register(&fp_dev);
}

static void __exit AGILE_GPU_exit(void)
{
    misc_deregister(&fp_dev);
    pr_info("%s Exit\n", DEVICE_NAME);
}

module_init(AGILE_GPU_init);
module_exit(AGILE_GPU_exit);
// #endif