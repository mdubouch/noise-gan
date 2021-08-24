
// The files this reads will need a title in the format "Af1_Af2_Af3_ 'image_filename' "
// The variables do not have to be called "Analyser, Stat" etc. in this code, but they must not be called Parameter. 
// 	Some other such entries may also glitch if they appear in other local plotify scripts. 
//	The variable must also be used as its name: Var = new Parameter( 'Var', ...)

// Parameters
// These are the clickable parameter options that will show on the side of the page - this changes pages and visible images
//	They contain the visible title "Visible ... A", and a part of the file name that relates to this parameter "Af1"
//	Any number of files may be called to "Af1", "Bf1", etc.

job_id = new Parameter('job_id', 
    job_id_list,
    job_id_list
);

// Image config
// List of filename segments of images that will appear
var image_name_list = [
        "losses",
        "comp_doca",
        "comp_edep",
        "comp_t",
        "comp_edep_per_layer",
        //"comp_edep_per_wire",
        "comp_layer",
        "comp_scatter",
        "comp_theta",
        "comp_wire",
        "gen_edep",
        "feature_matrix_fake",
        "feature_matrix_real",
        "real_scatter",
        "gen_scatter",
        "grid_real",
        "grid_fake",
        "comp_dist",
        "comp_time_diff",
        "activated_wires",
        "gp",
        "critic_score",
        "ae_losses",
        "pretrain_loss",
        "pretrain_acc",
        "pretrain_dist_loss",
        "dist_mean_var_losses",
];

// Compiles the filename and loops through the individual images on a page. The following is one storage structure example
image_filename = function(image_name){ 
    return ["output_", job_id, '/', image_name];
}

// Should show up as the following, with the default images coming from the first of each Parameter() option:
//
//        Page Title (if present in html file)
// 
// Analyser  |      image_1     image_2
// Stat      |
// Particle  |      image_3     image_4
//           |
//


var image_extension = "png";
var image_download = "png";
var n_cols = 4; // number of columns of images. May likely also require adapting in 'index.html' under  "<td colspan=2>"

