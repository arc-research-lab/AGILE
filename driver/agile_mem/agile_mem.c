#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>
#include <linux/ioctl.h>
#include <linux/io.h>

#define DEVICE_NAME "agile_mem"
#define RESERVED_MEM_IOCTL_GET_PHYS _IOR('R', 1, unsigned long)

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zhuoping Yang");
MODULE_DESCRIPTION("Map reserved physical memory to userspace and expose physical address");
MODULE_VERSION("1.0");

// Allow users to pass GB directly via module parameters
static unsigned long reserved_offset_gb = 128;
static unsigned long reserved_size_gb = 64;
module_param(reserved_offset_gb, ulong, 0444);
module_param(reserved_size_gb, ulong, 0444);
MODULE_PARM_DESC(reserved_offset_gb, "Offset of reserved physical memory in GB");
MODULE_PARM_DESC(reserved_size_gb, "Size of reserved memory in GB");

static unsigned long reserved_phys_offset;
static unsigned long reserved_mem_size;

static int reserved_mem_mmap(struct file *filp, struct vm_area_struct *vma)
{
    unsigned long size = vma->vm_end - vma->vm_start;

    if (size > reserved_mem_size) {
        pr_err("[reserved_mem] mmap size too large: %lu > %lu\n", size, reserved_mem_size);
        return -EINVAL;
    }

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    if (remap_pfn_range(vma, vma->vm_start,
                        reserved_phys_offset >> PAGE_SHIFT,
                        size, vma->vm_page_prot)) {
        pr_err("[reserved_mem] remap_pfn_range failed\n");
        return -EAGAIN;
    }

    pr_info("[reserved_mem] Memory mapped: size = %lu\n", size);
    return 0;
}

static long reserved_mem_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    if (cmd == RESERVED_MEM_IOCTL_GET_PHYS) {
        return copy_to_user((unsigned long __user *)arg,
                            &reserved_phys_offset,
                            sizeof(unsigned long)) ? -EFAULT : 0;
    }
    return -ENOTTY;
}

static const struct file_operations reserved_mem_fops = {
    .owner = THIS_MODULE,
    .mmap = reserved_mem_mmap,
    .unlocked_ioctl = reserved_mem_ioctl,
};

static struct miscdevice reserved_mem_dev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = DEVICE_NAME,
    .fops = &reserved_mem_fops,
    .mode = 0666,
};

static int __init reserved_mem_init(void)
{
    reserved_phys_offset = reserved_offset_gb << 30;
    reserved_mem_size = reserved_size_gb << 30;

    pr_info("[reserved_mem] Init: phys=0x%lx size=%lu bytes\n",
            reserved_phys_offset, reserved_mem_size);

    return misc_register(&reserved_mem_dev);
}

static void __exit reserved_mem_exit(void)
{
    misc_deregister(&reserved_mem_dev);
    pr_info("[reserved_mem] Exit\n");
}

module_init(reserved_mem_init);
module_exit(reserved_mem_exit);
