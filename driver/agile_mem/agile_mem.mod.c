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
	{ 0x9060573, "param_ops_ulong" },
	{ 0x6ad75c9d, "misc_deregister" },
	{ 0xc6241c0d, "misc_register" },
	{ 0xc5850110, "printk" },
	{ 0xb30f29d6, "remap_pfn_range" },
	{ 0xb665f56d, "__cachemode2pte_tbl" },
	{ 0xeb8461e2, "boot_cpu_data" },
	{ 0xb44ad4b3, "_copy_to_user" },
	{ 0xbdfb6dbb, "__fentry__" },
};

MODULE_INFO(depends, "");


MODULE_INFO(srcversion, "9F659F1C842F8407D27812D");
