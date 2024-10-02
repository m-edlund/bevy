//! Demonstrates different sub view effects.
//!
//! A sub view is essentially a smaller section of a larger viewport. Some use
//! cases include:
//! - Split one image across multiple cameras, for use in a multimonitor setups
//! - Magnify a section of the image, by rendering a small sub view in another
//!   camera
//! - Rapidly change the sub view offset to get a screen shake effect
use bevy::{
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::camera::{SubCameraView, Viewport},
};
use bevy_render::{
    camera::ScalingMode,
    render_resource::{AsBindGroup, ShaderRef, ShaderType},
};

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/camera_sub_view.wgsl";

const SMALL_SIZE: u32 = 200;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MaterialPlugin::<
            ExtendedMaterial<StandardMaterial, InvertMaterialExtension>,
        >::default())
        .init_resource::<AspectRatioDivider>()
        .init_resource::<Offsetter>()
        .add_systems(Startup, setup)
        .add_systems(Update, (move_camera_view, change_settings))
        .run();
}

#[derive(Debug, Component)]
struct MovingCameraMarker;

/// Set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials_extended: ResMut<
        Assets<ExtendedMaterial<StandardMaterial, InvertMaterialExtension>>,
    >,
) {
    let transform = Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::ZERO, Vec3::Y);

    let main_camera_viewport = ViewportWithPadding {
        main_camera_viewport: Vec2::new(1200.0, 720.0),
        _webgl2_padding: Vec2::ZERO,
    };

    // Plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(5.0, 5.0))),
        MeshMaterial3d(materials_extended.add(ExtendedMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.3, 0.5, 0.3),
                ..Default::default()
            },
            extension: InvertMaterialExtension {
                main_camera_viewport,
            },
        })),
    ));

    // Cube
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::default())),
        MeshMaterial3d(materials_extended.add(ExtendedMaterial {
            base: StandardMaterial {
                base_color: Color::srgb(0.8, 0.7, 0.6),
                ..Default::default()
            },
            extension: InvertMaterialExtension {
                main_camera_viewport,
            },
        })),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    // Light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // Main perspective Camera
    commands.spawn(Camera3dBundle {
        transform,
        ..default()
    });

    // Perspective camera moving
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                viewport: Option::from(Viewport {
                    physical_size: UVec2::new(SMALL_SIZE, SMALL_SIZE),
                    ..default()
                }),
                sub_camera_view: Some(SubCameraView {
                    // Set the sub view camera to a fifth of the full view and
                    // move it in another system
                    full_size: UVec2::new(500, 500),
                    offset: Vec2::ZERO,
                    size: UVec2::new(SMALL_SIZE, SMALL_SIZE),
                }),
                order: 2,
                ..default()
            },
            transform,
            ..default()
        },
        MovingCameraMarker,
    ));
}

fn move_camera_view(
    main_camera_query: Query<&mut Camera, Without<MovingCameraMarker>>,
    mut movable_camera_query: Query<&mut Camera, With<MovingCameraMarker>>,
    mut materials_extended: ResMut<
        Assets<ExtendedMaterial<StandardMaterial, InvertMaterialExtension>>,
    >,
    material_handles: Query<&Handle<ExtendedMaterial<StandardMaterial, InvertMaterialExtension>>>,
    aspect_ratio_settings: Res<AspectRatioDivider>,
    offset_settings: Res<Offsetter>,
) {
    let main_camera = main_camera_query.single();

    let UVec2 {
        x: width,
        y: height,
    } = main_camera.physical_viewport_size().unwrap();

    for material_handle in material_handles.iter() {
        let material = materials_extended.get_mut(material_handle).unwrap();
        material.extension.main_camera_viewport = ViewportWithPadding {
            main_camera_viewport: Vec2::new(width as f32, height as f32),
            _webgl2_padding: Vec2::ZERO,
        };
    }

    let UVec2 { x: div_x, y: div_y } = aspect_ratio_settings.0;
    let offset = offset_settings.0;

    let new_size = UVec2::new(width / div_x, height / div_y)
        .max(UVec2::ZERO)
        .min(UVec2::new(
            width.max(offset.x) - offset.x,
            height.max(offset.y) - offset.y,
        ));

    for mut camera in movable_camera_query.iter_mut() {
        if let Some(sub_view) = &mut camera.sub_camera_view {
            sub_view.full_size = UVec2::new(width, height);
            sub_view.size = new_size;
            sub_view.offset = Vec2::new(offset.x as f32, offset.y as f32);
        }

        if let Some(viewport) = &mut camera.viewport {
            viewport.physical_position = offset;
            viewport.physical_size = new_size;
        }
    }
}

fn change_settings(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut aspect_ratio: ResMut<AspectRatioDivider>,
    mut offset: ResMut<Offsetter>,
    mut camera_query: Query<&mut Projection>,
) {
    // Aspect ratio
    if keyboard.just_pressed(KeyCode::KeyD) && aspect_ratio.0.x > 2 {
        aspect_ratio.0.x -= 1;
    }
    if keyboard.just_pressed(KeyCode::KeyA) {
        aspect_ratio.0.x += 1;
    }
    if keyboard.just_pressed(KeyCode::KeyS) {
        aspect_ratio.0.y += 1;
    }
    if keyboard.just_pressed(KeyCode::KeyW) && aspect_ratio.0.y > 2 {
        aspect_ratio.0.y -= 1;
    }

    // Offset
    if keyboard.pressed(KeyCode::ArrowRight) {
        offset.0.x += 10;
    }
    if keyboard.pressed(KeyCode::ArrowLeft) && offset.0.x > 10 {
        offset.0.x -= 10;
    }
    if keyboard.pressed(KeyCode::ArrowUp) && offset.0.y > 10 {
        offset.0.y -= 10;
    }
    if keyboard.pressed(KeyCode::ArrowDown) {
        offset.0.y += 10;
    }

    if keyboard.just_pressed(KeyCode::KeyP) {
        for mut projection in camera_query.iter_mut() {
            *projection = match *projection {
                Projection::Perspective(_) => OrthographicProjection {
                    // 6 world units per window height.
                    scaling_mode: ScalingMode::FixedVertical(6.0),
                    ..OrthographicProjection::default_3d()
                }
                .into(),
                Projection::Orthographic(_) => PerspectiveProjection::default().into(),
            }
        }
    };
}

#[derive(Debug, Resource)]
struct AspectRatioDivider(UVec2);

impl Default for AspectRatioDivider {
    fn default() -> Self {
        Self(UVec2::new(2, 2))
    }
}

#[derive(Debug, Resource, Default)]
struct Offsetter(UVec2);

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
struct InvertMaterialExtension {
    #[uniform(102)]
    main_camera_viewport: ViewportWithPadding,
}

#[derive(Copy, Clone, ShaderType, Debug)]
struct ViewportWithPadding {
    main_camera_viewport: Vec2,
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: Vec2,
}

impl MaterialExtension for InvertMaterialExtension {
    fn fragment_shader() -> ShaderRef {
        SHADER_ASSET_PATH.into()
    }
}

// fn move_camera(
//     keyboard_input: Res<ButtonInput<KeyCode>>,
//     mut query: Query<&mut Transform, With<Camera>>,
//     main_camera_query: Query<&mut Camera, Without<MovingCameraMarker>>,
//     mut movable_camera_query: Query<&mut Camera, With<MovingCameraMarker>>,
// ) {
//     // Move
//     if keyboard_input.just_pressed(KeyCode::ArrowLeft) {
//         transform.translation.x -= move_amount;
//     }
//     if keyboard_input.just_pressed(KeyCode::ArrowRight) {
//         transform.translation.x += move_amount;
//     }
//     if keyboard_input.just_pressed(KeyCode::ArrowUp) {
//         transform.translation.y += move_amount;
//     }
//     if keyboard_input.just_pressed(KeyCode::ArrowDown) {
//         transform.translation.y -= move_amount;
//     }
// }
