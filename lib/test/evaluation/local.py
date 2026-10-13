from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data1/wangwj/CADTrack-main/got10k_lmdb'
    settings.got10k_path = '/data1/wangwj/CADTrack-main/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data1/wangwj/CADTrack-main/itb'
    settings.lasot_extension_subset_path_path = '/data1/wangwj/CADTrack-main/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data1/wangwj/CADTrack-main/lasot_lmdb'
    settings.lasot_path = '/data1/wangwj/CADTrack-main/lasot'
    settings.network_path = '/data1/wangwj/CADTrack-main/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data1/wangwj/CADTrack-main/nfs'
    settings.otb_path = '/data1/wangwj/CADTrack-main/otb'
    settings.prj_dir = '/data3/wangwj/CADTrack-main'
    settings.result_plot_path = '/data2/wangwj/CADTrack-main/baseline/result_plots'
    settings.results_path = '/data2/wangwj/CADTrack-main/baseline/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data3/wangwj/CADTrack-main/baseline'
    settings.segmentation_path = '/data1/wangwj/CADTrack-main/test/segmentation_results'
    settings.tc128_path = '/data1/wangwj/CADTrack-main/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data1/wangwj/CADTrack-main/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data1/wangwj/CADTrack-main/trackingnet'
    settings.uav_path = '/data1/wangwj/CADTrack-main/uav'
    settings.vot18_path = '/data1/wangwj/CADTrack-main/vot2018'
    settings.vot22_path = '/data1/wangwj/CADTrack-main/vot2022'
    settings.vot_path = '/data1/wangwj/CADTrack-main/VOT2019'
    settings.youtubevos_dir = ''

    return settings

