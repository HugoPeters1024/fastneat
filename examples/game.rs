use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use genome::{
    ctrnn::Ctrnn,
    params::{ActivationFunction, Parameters, Settings},
    population::Population,
};
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, handle_reset)
        .add_systems(FixedUpdate, handle_agents)
        .add_event::<ResetEvent>()
        .run();
}

#[derive(Resource)]
struct Neat {
    pop: Population,
    ticks: usize,
    agents: Vec<Entity>,
}

#[derive(Component)]
struct Brain(Ctrnn);

#[derive(Component)]
struct Target;

#[derive(Event)]
struct ResetEvent;

const SIMULATION_SPEED: f32 = 10.0;

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut events: EventWriter<ResetEvent>,
    mut rapier_config: ResMut<RapierConfiguration>,
) {
    rapier_config.timestep_mode = TimestepMode::Fixed {
        dt: SIMULATION_SPEED / 64.0,
        substeps: SIMULATION_SPEED as usize,
    };
    commands.insert_resource(Time::<Fixed>::from_hz(SIMULATION_SPEED as f64 * 64.0));

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Plane3d::default().mesh().size(20., 20.)),
            material: materials.add(Color::WHITE),
            transform: Transform::default(),
            ..default()
        },
        Collider::cuboid(10.0, 0.1, 10.0),
        CollisionGroups::new(Group::GROUP_1, Group::GROUP_2),
    ));

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 24.5, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
            material: materials.add(Color::srgb_u8(62, 255, 88)),
            transform: Transform::from_translation(random_target()),
            ..default()
        },
        Target,
    ));

    commands.insert_resource(Neat {
        pop: Population::new(&Settings {
            num_inputs: 2,
            num_outputs: 2,
            population_size: 200,
            target_species: 10,
            parameters: Parameters {
                specie_greediness_exponent: 1.5,
                start_with_bias_connections: false,
                specie_threshold_nudge_factor: 0.5,
                activation_function: ActivationFunction::Tanh,
                ..default()
            },
        }),
        ticks: 0,
        agents: Vec::new(),
    });

    events.send(ResetEvent);
}

fn random_target() -> Vec3 {
    let mut rng = rand::thread_rng();
    const PI: f32 = 3.1415926536;
    let t = rng.gen::<f32>() * 2.0 * PI;
    let x = t.cos() * 8.0;
    let y = 0.5;
    let z = t.sin() * 8.0;
    Vec3::new(x, y, z)
}

fn handle_reset(
    mut commands: Commands,
    mut reader: EventReader<ResetEvent>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut neat: ResMut<Neat>,
    mut target: Query<&mut Transform, With<Target>>,
) {
    for _ in reader.read() {
        for agent in &neat.agents {
            commands.entity(*agent).despawn_recursive();
        }

        neat.agents.clear();
        let mut agents = Vec::new();
        for member in &neat.pop.members {
            agents.push(
                commands
                    .spawn((
                        Brain(neat.pop.get_phenotype(member)),
                        PbrBundle {
                            mesh: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
                            material: materials.add(Color::srgb_u8(124, 144, 255)),
                            transform: Transform::from_xyz(0.0, 0.5, 0.0),
                            ..default()
                        },
                        Collider::cuboid(0.5, 0.5, 0.5),
                        RigidBody::Dynamic,
                        ExternalForce::default(),
                        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
                    ))
                    .id(),
            )
        }
        neat.agents = agents;
        target.single_mut().translation = random_target();
    }
}

fn handle_agents(
    mut writer: EventWriter<ResetEvent>,
    time: Res<Time>,
    mut q: Query<(&mut Brain, &mut ExternalForce, &Transform)>,
    mut target: Query<&mut Transform, (With<Target>, Without<Brain>)>,
    mut neat: ResMut<Neat>,
) {
    let target = &mut target.single_mut().translation;
    for (mut brain, mut forces, transform) in q.iter_mut() {
        brain.0.update(
            time.delta_seconds_f64(),
            &[
                (target.x as f64 - transform.translation.x as f64) / 15.0,
                (target.z as f64 - transform.translation.z as f64) / 15.0,
            ],
        );
        let outputs = brain.0.get_outputs();
        forces.force.x = 6.0 * outputs[0] as f32;
        forces.force.z = 6.0 * outputs[1] as f32;
    }

    const SIMULATION_LENGTH: usize = (70.0 * SIMULATION_SPEED) as usize;

    neat.ticks += 1;
    if neat.ticks % SIMULATION_LENGTH == 0 {
        for (agent_idx, agent_entity) in neat.agents.clone().iter().cloned().enumerate() {
            let transform = q.get(agent_entity).unwrap().2;
            neat.pop.members[agent_idx].fitness +=
                (40.0 - transform.translation.distance(*target) as f64).max(0.0);
        }
        if neat.ticks % (5 * SIMULATION_LENGTH) == 0 {
            neat.pop.evolve();
            neat.ticks = 0;
        }
        writer.send(ResetEvent);
    }
}
