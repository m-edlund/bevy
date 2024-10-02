#import bevy_pbr::{
    mesh_view_bindings as view_bindings,
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}

@group(2) @binding(102) var<uniform> main_camera_viewport: MainCameraViewportWithPadding;

struct MainCameraViewportWithPadding {
    viewport: vec2<f32>,
    _webgl2_padding: vec2<f32>
}

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    let width = view_bindings::view.viewport.z;
    let height = view_bindings::view.viewport.w;

    if (width < main_camera_viewport.viewport.x)  {
        pbr_input.material.base_color.r = 1.0 - pbr_input.material.base_color.r;
        pbr_input.material.base_color.g = 1.0 - pbr_input.material.base_color.g;
        pbr_input.material.base_color.b = 1.0 - pbr_input.material.base_color.b;
    }

    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);

    return out;
}
