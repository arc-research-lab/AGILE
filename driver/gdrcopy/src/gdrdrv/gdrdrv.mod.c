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
	{ 0xdd3af736, "param_ops_int" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0xac549b65, "__register_chrdev" },
	{ 0xdecd0b29, "__stack_chk_fail" },
	{ 0x970adefe, "nvidia_p2p_get_pages_persistent" },
	{ 0x318d6fec, "mutex_is_locked" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0xd6b33026, "cpu_khz" },
	{ 0x5b3f3e79, "nvidia_p2p_get_pages" },
	{ 0x7b4da6ff, "__init_rwsem" },
	{ 0xd6ee688f, "vmalloc" },
	{ 0xb44ad4b3, "_copy_to_user" },
	{ 0x53b954a2, "up_read" },
	{ 0x668b19a1, "down_read" },
	{ 0x362ef408, "_copy_from_user" },
	{ 0xb30f29d6, "remap_pfn_range" },
	{ 0x8a35b432, "sme_me_mask" },
	{ 0xf42ca687, "nvidia_p2p_free_page_table" },
	{ 0x567bf9df, "address_space_init_once" },
	{ 0xf8aae665, "current_task" },
	{ 0x977f511b, "__mutex_init" },
	{ 0xc622a0d1, "kmem_cache_alloc_trace" },
	{ 0xe94c7a3a, "kmalloc_caches" },
	{ 0x409bcb62, "mutex_unlock" },
	{ 0x57bc19d2, "down_write" },
	{ 0x2ab7989d, "mutex_lock" },
	{ 0xacdf3914, "nvidia_p2p_put_pages_persistent" },
	{ 0x37a0cba, "kfree" },
	{ 0x642487ac, "nvidia_p2p_put_pages" },
	{ 0xce807a25, "up_write" },
	{ 0x911cad6a, "unmap_mapping_range" },
	{ 0xc5850110, "printk" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "nv-p2p-dummy");


MODULE_INFO(srcversion, "29B26B7E8ACB3B02AD9594A");
