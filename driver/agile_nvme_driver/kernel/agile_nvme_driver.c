// SPDX-License-Identifier: GPL-2.0
#include <linux/module.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/list.h>
#include <linux/slab.h>

#include "../common/agile_nvme_driver.h"

#define DRIVER_NAME         "AGILE NVMe SSD Driver"
#define DEVICE_NAME_FMT     "AGILE-NVMe-%s"
#define CLASS_NAME          "AGILE Class"

#define BAR_MMAP            0
#define DMA_MMAP            1

struct nvme_dev {
    struct pci_dev      *pdev;
    struct device       *device;
    struct cdev         cdev;
    dev_t               devt;
    uint32_t            index;
    uint32_t            mmap_flag;
    struct dma_buffer   dma_buf;
    struct bar_info     bar;
    struct list_head    list;
};

static struct class *nvme_class;
static LIST_HEAD(nvme_device_list);

static long nvme_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct nvme_dev *ndev = file->private_data;
    switch(cmd){
        case IOCTL_GET_BAR:
            if (copy_to_user((void __user *)arg, &ndev->bar, sizeof(ndev->bar)))
                return -EFAULT;

            break;
        case IOCTL_ALLOCATE_DMA_BUFFER:
            if (copy_from_user(&ndev->dma_buf, (void __user *)arg, sizeof(ndev->dma_buf)))
                return -EFAULT;

            if (ndev->dma_buf.size == 0) {
                pr_err("%s: DMA buffer size cannot be zero\n", DRIVER_NAME);
                return -EINVAL;
            }

            // align size to page size
            ndev->dma_buf.size = PAGE_ALIGN(ndev->dma_buf.size);

            ndev->dma_buf.vaddr = dma_alloc_coherent(&ndev->pdev->dev,
                                                      ndev->dma_buf.size,
                                                      &ndev->dma_buf.addr,
                                                      GFP_KERNEL);
            if (!ndev->dma_buf.vaddr) {
                pr_err("%s: Failed to allocate DMA buffer\n", DRIVER_NAME);
                return -ENOMEM;
            }

            if (copy_to_user((void __user *)arg, &ndev->dma_buf, sizeof(ndev->dma_buf)))
                return -EFAULT;

            ndev->mmap_flag = DMA_MMAP;

            pr_info("%s: Allocated DMA buffer at addr=0x%llx size=%u\n",
                    DRIVER_NAME, (unsigned long long)ndev->dma_buf.addr, ndev->dma_buf.size);

            break;
        case IOCTL_FREE_DMA_BUFFER:
            if (ndev->dma_buf.vaddr) {
                dma_free_coherent(&ndev->pdev->dev, ndev->dma_buf.size,
                                  ndev->dma_buf.vaddr, ndev->dma_buf.addr);
                ndev->dma_buf.vaddr = NULL;
                ndev->dma_buf.addr = 0;
                ndev->dma_buf.size = 0;
                pr_info("%s: Freed DMA buffer\n", DRIVER_NAME);
            } else {
                pr_err("%s: No DMA buffer allocated to free\n", DRIVER_NAME);
                return -EINVAL;
            }

            if (copy_to_user((void __user *)arg, &ndev->dma_buf, sizeof(ndev->dma_buf)))
                return -EFAULT;

            break;
        case IOCTL_SET_DMA_BUFFER:
            if (copy_from_user(&ndev->dma_buf, (void __user *)arg, sizeof(ndev->dma_buf)))
                return -EFAULT;

            if (ndev->dma_buf.size == 0) {
                pr_err("%s: DMA buffer size cannot be zero\n", DRIVER_NAME);
                return -EINVAL;
            }
            
            break;
        case IOCTL_SET_MMAP_TO_BAR:
            ndev->mmap_flag = BAR_MMAP;
            break;
        case IOCTL_SET_MMAP_TO_DMA:
            ndev->mmap_flag = DMA_MMAP;
            break;
        // TEST FUNCTIONS
        case IOCTL_QUERY_MMAP_FLAG:
            if (copy_to_user((void __user *)arg, &ndev->mmap_flag, sizeof(ndev->mmap_flag)))
                return -EFAULT;
            break;
        default:
            pr_err("%s: IOCTL command not supported: %d\n", DRIVER_NAME, cmd);
            return -EFAULT;
    }
    return 0;
}

static int nvme_open(struct inode *inode, struct file *file)
{
    struct nvme_dev *ndev = container_of(inode->i_cdev, struct nvme_dev, cdev);
    ndev->mmap_flag = BAR_MMAP;
    file->private_data = ndev;
    pr_info("%s: PCI device = %s open\n", DRIVER_NAME, pci_name(ndev->pdev));
    return 0;
}

static int fp_mmap(struct file *filp, struct vm_area_struct *vma)
{
    struct nvme_dev *ndev = filp->private_data;
    unsigned long req_size = vma->vm_end - vma->vm_start;

    switch(ndev->mmap_flag){
        case BAR_MMAP:
            if (req_size > ndev->bar.size){
                pr_err("%s: requested mmap size is larger than the avaliable BAR size\n", DRIVER_NAME);
                return -EINVAL;
            }
            // Optional: make mapping non-cached for MMIO
            vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
            pr_info("%s: remapping BAR at addr=0x%llx size=%llu\n",
                    DRIVER_NAME, (unsigned long long)ndev->bar.phys_addr, (unsigned long long)ndev->bar.size);
            return remap_pfn_range(vma,
                            vma->vm_start,
                            ndev->bar.phys_addr >> PAGE_SHIFT,
                            req_size,
                            vma->vm_page_prot);
        case DMA_MMAP:
            if (req_size > ndev->dma_buf.size){
                pr_err("%s: requested mmap size is larger than the avaliable dma buffer size %ld > %d addr: 0x%llx\n",
                        DRIVER_NAME, req_size, ndev->dma_buf.size, (unsigned long long)ndev->dma_buf.addr);
                return -EINVAL;
            }

            if (!ndev->dma_buf.vaddr) {
                pr_err("%s: DMA buffer not allocated\n", DRIVER_NAME);
                return -EINVAL;
            }
            // Optional: make mapping non-cached for MMIO
            vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

            pr_info("%s: remapping DMA buffer at addr=0x%llx size=%u\n",
                    DRIVER_NAME, (unsigned long long)ndev->dma_buf.addr, ndev->dma_buf.size);
            return remap_pfn_range(vma,
                            vma->vm_start,
                            ndev->dma_buf.addr >> PAGE_SHIFT,
                            req_size,
                            vma->vm_page_prot);
    }

    pr_err("unknown mmap flag: %d\n", ndev->mmap_flag);
    return -EFAULT;
}

static const struct file_operations nvme_fops = {
    .owner = THIS_MODULE,
    .open = nvme_open,
    .unlocked_ioctl = nvme_ioctl,
    .mmap = fp_mmap,
};

static const struct pci_device_id nvme_ids[] = {
    { PCI_DEVICE(0x0c51, 0x0110) }, // Replace with real IDs as needed
    { 0, }
};

MODULE_DEVICE_TABLE(pci, nvme_ids);

static int nvme_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct nvme_dev *ndev;
    int bar = 0, ret;

    ndev = kzalloc(sizeof(*ndev), GFP_KERNEL);
    if (!ndev)
        return -ENOMEM;

    if (pci_enable_device(pdev)) {
        kfree(ndev);
        return -ENODEV;
    }

    ndev->bar.phys_addr = pci_resource_start(pdev, bar);
    ndev->bar.size = pci_resource_len(pdev, bar);

    ndev->pdev = pdev;
    pci_set_drvdata(pdev, ndev);

    ret = alloc_chrdev_region(&ndev->devt, 0, 1, DEVICE_NAME_FMT);
    if (ret)
        goto err_disable;

    cdev_init(&ndev->cdev, &nvme_fops);
    ndev->cdev.owner = THIS_MODULE;
    ret = cdev_add(&ndev->cdev, ndev->devt, 1);
    if (ret)
        goto err_unregister;

    ndev->device = device_create(nvme_class, NULL, ndev->devt, NULL, DEVICE_NAME_FMT, pci_name(pdev));
    if (IS_ERR(ndev->device)) {
        ret = PTR_ERR(ndev->device);
        goto err_cdev;
    }

    INIT_LIST_HEAD(&ndev->list);
    list_add_tail(&ndev->list, &nvme_device_list);

    pr_info("%s: registered %s BAR0=0x%llx size=0x%llx\n", DRIVER_NAME,
            pci_name(pdev), (unsigned long long)ndev->bar.phys_addr,
            (unsigned long long)ndev->bar.size);
    return 0;

err_cdev:
    cdev_del(&ndev->cdev);
err_unregister:
    unregister_chrdev_region(ndev->devt, 1);
err_disable:
    pci_disable_device(pdev);
    kfree(ndev);
    return ret;
}

static void nvme_remove(struct pci_dev *pdev)
{
    struct nvme_dev *ndev = pci_get_drvdata(pdev);

    device_destroy(nvme_class, ndev->devt);
    cdev_del(&ndev->cdev);
    unregister_chrdev_region(ndev->devt, 1);
    list_del(&ndev->list);
    pci_disable_device(pdev);
    kfree(ndev);
    pr_info("%s: removed device %s\n", DRIVER_NAME, pci_name(pdev));
}

static struct pci_driver nvme_driver = {
    .name = DRIVER_NAME,
    .id_table = nvme_ids,
    .probe = nvme_probe,
    .remove = nvme_remove,
};

static int __init nvme_init(void)
{
    nvme_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(nvme_class)) {
        return PTR_ERR(nvme_class);
    }

    return pci_register_driver(&nvme_driver);
}

static void __exit nvme_exit(void)
{
    struct nvme_dev *ndev, *tmp;

    pci_unregister_driver(&nvme_driver);

    list_for_each_entry_safe(ndev, tmp, &nvme_device_list, list) {
        list_del(&ndev->list);
        kfree(ndev);
    }
    class_destroy(nvme_class);
}

module_init(nvme_init);
module_exit(nvme_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Zhuoping Yang");
MODULE_DESCRIPTION("AGILE NVMe SSD Driver for accessing PCIe BAR and allocating and managing DMA buffers");