//! Renders a 2D scene containing a single, moving sprite.

use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};

fn main() {
    std::env::set_var("CARGO_MANIFEST_DIR", "examples");
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, move_player)
        .add_systems(Update, jump)
        .run();
}

#[derive(Component)]
struct Player {
    v_speed: f32,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut time: ResMut<Time<Virtual>>
) {
    time.set_relative_speed(1.0);
    commands.insert_resource(ClearColor(Color::srgb(0.2, 0.2, 0.2)));
    commands.spawn(Camera2dBundle::default());
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: Mesh2dHandle(meshes.add(Rectangle::new(64.0, 64.0))),
            material: materials.add(Color::srgb(0.8, 0.8, 0.8)),
            ..default()
        },
        Player { v_speed: 0.0 },
    ));
}

fn move_player(mut q: Query<(&mut Player, &mut Transform)>) {
    let (mut player, mut playert) = q.single_mut();
    playert.translation.y -= player.v_speed;
    player.v_speed += 1.2;
    if playert.translation.y < -200.0 {
        playert.translation.y = -200.0;
        player.v_speed = 0.0;
    }
}

fn jump(mut q: Query<(&mut Player, &Transform)>, keyboard: Res<ButtonInput<KeyCode>>) {
    let (mut player, playert) = q.single_mut();
    if keyboard.just_pressed(KeyCode::Space) && playert.translation.y <= -200.0 {
        player.v_speed = -25.0;
    }
}
