#include <linux/build-salt.h>
#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(.gnu.linkonce.this_module) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section(__versions) = {
	{ 0xe4c970fb, "module_layout" },
	{ 0x201d0f70, "class_destroy" },
	{ 0x6446db14, "pci_unregister_driver" },
	{ 0x2f02ac15, "__pci_register_driver" },
	{ 0xa640d12, "__class_create" },
	{ 0x241c4110, "device_create" },
	{ 0x47cb61f7, "cdev_add" },
	{  0x22a98, "cdev_init" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0x4eb16ccf, "pci_enable_device" },
	{ 0xc622a0d1, "kmem_cache_alloc_trace" },
	{ 0xe94c7a3a, "kmalloc_caches" },
	{ 0x9f04a881, "dma_free_attrs" },
	{ 0xe12fe387, "dma_alloc_attrs" },
	{ 0xb44ad4b3, "_copy_to_user" },
	{ 0x362ef408, "_copy_from_user" },
	{ 0xb30f29d6, "remap_pfn_range" },
	{ 0xc5850110, "printk" },
	{ 0x37a0cba, "kfree" },
	{ 0x80daf046, "pci_disable_device" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0x60e2055d, "cdev_del" },
	{ 0x260bbe45, "device_destroy" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "");

MODULE_ALIAS("pci:v00000C51d00000110sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000144Dd0000A80Csv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000144Dd0000A824sv*sd*bc*sc*i*");

MODULE_INFO(srcversion, "2DC529C0159947484F3F5F1");
