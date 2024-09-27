//! Renders a 2D scene containing a single, moving sprite.

use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_editor_pls::prelude::*;
use fastneat::{ctrnn::Ctrnn, params::Settings, population::Population};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EditorPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, render_game)
        .add_systems(FixedUpdate, tick_games)
        .run();
}

#[derive(Resource)]
struct AllAssets {
    rect_mesh: Mesh2dHandle,
}

struct Piece {
    kind: usize,
    x: isize,
    y: isize,
    rot: isize,
}

impl Piece {
    fn get_rot(&self) -> usize {
        self.rot.rem_euclid(4) as usize
    }
}

#[derive(Component)]
enum Controller {
    Human,
    AI(Ctrnn),
}

#[derive(Component)]
struct Game {
    board: Vec<bool>,
    width: usize,
    height: usize,
    current_piece: Piece,
}

#[derive(Clone, Copy)]
enum MoveInstr {
    GoLeft,
    GoRight,
}

#[derive(Clone, Copy)]
enum RotateInstr {
    Clockwise,
    Counterwise,
}

impl Game {
    pub fn new(width: usize, height: usize) -> Self {
        Game {
            board: vec![false; width * height],
            width,
            height,
            current_piece: Piece {
                kind: 0,
                x: 0,
                y: 0,
                rot: 0,
            },
        }
    }

    pub fn get(&self, x: isize, y: isize) -> bool {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return true;
        }
        let x = x as usize;
        let y = y as usize;
        self.board[self.width * y + x]
    }

    pub fn set(&mut self, x: isize, y: isize) {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return;
        }
        let x = x as usize;
        let y = y as usize;
        self.board[self.width * y + x] = true;
    }

    pub fn in_conflict(&self) -> bool {
        for dy in 0..3 {
            for dx in 0..3 {
                let x = self.current_piece.x + dx as isize;
                let y = self.current_piece.y + dy as isize;
                if PIECES[self.current_piece.kind][self.current_piece.get_rot()][dy * 3 + dx] == '#'
                    && self.get(x, y)
                {
                    return true;
                }
            }
        }

        return false;
    }

    pub fn tick_input(&mut self, move_instr: Option<MoveInstr>, rotate_instr: Option<RotateInstr>) {
        let horz_move: isize = match move_instr {
            None => 0,
            Some(MoveInstr::GoRight) => 1,
            Some(MoveInstr::GoLeft) => -1,
        };

        self.current_piece.x += horz_move;
        if self.in_conflict() {
            self.current_piece.x -= horz_move;
        }

        let rot_move: isize = match rotate_instr {
            None => 0,
            Some(RotateInstr::Clockwise) => 1,
            Some(RotateInstr::Counterwise) => -1,
        };

        self.current_piece.rot += rot_move;
        if self.in_conflict() {
            self.current_piece.rot -= rot_move;
        }
    }

    pub fn tick_gravity(&mut self) {
        self.current_piece.y += 1;
        if self.in_conflict() {
            self.current_piece.y -= 1;
            for dy in 0..3 {
                for dx in 0..3 {
                    let x = self.current_piece.x + dx as isize;
                    let y = self.current_piece.y + dy as isize;
                    if PIECES[self.current_piece.kind][self.current_piece.get_rot()][dy * 3 + dx]
                        == '#'
                    {
                        self.set(x, y);
                    }
                }
            }

            self.current_piece.x = 0;
            self.current_piece.y = 0;
            self.current_piece.kind += 1;
            self.current_piece.kind %= PIECES.len();
        }
    }
}

#[derive(Resource)]
struct GameRender {
    board: Vec<Entity>,
    width: usize,
    height: usize,
    game_to_render: Option<Entity>,
}

impl GameRender {
    pub fn get_entity(&self, x: isize, y: isize) -> Option<Entity> {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return None;
        }
        let x = x as usize;
        let y = y as usize;
        return Some(self.board[self.width * y + x]);
    }
}

const BASE_PIECES: [[char; 9]; 7] = [
    ['.', '#', '.', '.', '#', '.', '.', '#', '.'], // i block
    ['#', '#', '.', '#', '#', '.', '.', '.', '.'], // o block
    ['.', '#', '.', '#', '#', '#', '.', '.', '.'], // t block
    ['.', '#', '.', '.', '#', '.', '.', '#', '#'], // l block
    ['.', '#', '.', '.', '#', '.', '#', '#', '.'], // j block
    ['.', '#', '#', '#', '#', '.', '.', '.', '.'], // s block
    ['#', '#', '.', '.', '#', '#', '.', '.', '.'], // z block
];

const PIECES: [[[char; 9]; 4]; 7] = [
    all_rotations(&BASE_PIECES[0]),
    all_rotations(&BASE_PIECES[1]),
    all_rotations(&BASE_PIECES[2]),
    all_rotations(&BASE_PIECES[3]),
    all_rotations(&BASE_PIECES[4]),
    all_rotations(&BASE_PIECES[5]),
    all_rotations(&BASE_PIECES[6]),
];

const fn all_rotations(piece: &[char; 9]) -> [[char; 9]; 4] {
    let mut ret = [['.'; 9]; 4];
    let mut piece = *piece;
    let mut i = 0;
    while i < 4 {
        ret[i] = piece;
        piece = rotate_clockwise(&piece);
        i += 1;
    }

    return ret;
}

const fn rotate_clockwise(piece: &[char; 9]) -> [char; 9] {
    [
        piece[6], piece[3], piece[0], // 0
        piece[7], piece[4], piece[1], // 1
        piece[8], piece[5], piece[2], // 3
    ]
}

#[derive(Resource)]
struct NeatState {
    pop: Population,
    agents: Vec<Entity>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle::default());

    let all_assets = AllAssets {
        rect_mesh: Mesh2dHandle(meshes.add(Rectangle::new(1.0, 1.0))),
    };

    let width = 10;
    let height = 20;

    let mut render = GameRender {
        board: Vec::new(),
        width,
        height,
        game_to_render: None,
    };

    commands
        .spawn((
            TransformBundle::from_transform(Transform::from_scale(Vec3::new(32.0, -32.0, 1.0))),
            InheritedVisibility::default(),
        ))
        .with_children(|parent| {
            for y in 0..height {
                for x in 0..width {
                    let color = Color::WHITE;

                    render.board.push(
                        parent
                            .spawn(MaterialMesh2dBundle {
                                mesh: all_assets.rect_mesh.clone(),
                                material: materials.add(color),
                                transform: Transform::from_xyz(x as f32, y as f32, 0.0),
                                ..default()
                            })
                            .id(),
                    );
                }
            }
        });

    commands.insert_resource(all_assets);

    const POP_SIZE: usize = 100;
    let pop = Population::new(&Settings {
        population_size: POP_SIZE,
        target_species: 1,
        num_inputs: width * height,
        num_outputs: 2,
        parameters: fastneat::params::Parameters {
            activation_function: fastneat::params::ActivationFunction::Tanh,
            ..default()
        },
    });

    let mut agents = Vec::new();
    for agent_idx in 0..POP_SIZE {
        let genome = &pop.members[agent_idx];
        let network = pop.get_phenotype(&genome);
        agents.push(
            commands
                .spawn((Game::new(width, height), Controller::AI(network)))
                .id(),
        );
    }

    render.game_to_render = Some(agents[0].clone());

    commands.insert_resource(NeatState { pop, agents });

    commands.insert_resource(render);
}

fn render_game(
    games: Query<&Game>,
    mut visibility: Query<&mut Visibility>,
    render: Res<GameRender>,
) {
    let Some(game_entity) = render.game_to_render else {
        return;
    };
    let game = games.get(game_entity).unwrap();
    assert!(game.width == render.width && game.height == render.height);
    for y in 0..game.height {
        for x in 0..game.width {
            let block_entity = render.board[y * game.width + x];
            *visibility.get_mut(block_entity).unwrap() = Visibility::Hidden;
            if game.get(x as isize, y as isize) {
                *visibility.get_mut(block_entity).unwrap() = Visibility::Visible
            }
        }
    }

    for dx in 0..3 {
        for dy in 0..3 {
            let x = game.current_piece.x + dx as isize;
            let y = game.current_piece.y + dy as isize;
            if PIECES[game.current_piece.kind][game.current_piece.get_rot()][dy * 3 + dx] == '#' {
                if let Some(block_entity) = render.get_entity(x, y) {
                    *visibility.get_mut(block_entity).unwrap() = Visibility::Visible
                }
            }
        }
    }
}

fn tick_games(
    mut games: Query<(&mut Game, &mut Controller)>,
    mut ticks: Local<usize>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut human_move_instr: Local<Option<MoveInstr>>,
    mut human_rotate_instr: Local<Option<RotateInstr>>,
    neat: Res<NeatState>,
) {
    *ticks += 1;

    if keyboard.just_pressed(KeyCode::ArrowRight) {
        *human_move_instr = Some(MoveInstr::GoRight);
    }
    if keyboard.just_pressed(KeyCode::ArrowLeft) {
        *human_move_instr = Some(MoveInstr::GoLeft);
    }
    if keyboard.just_pressed(KeyCode::KeyW) || keyboard.just_pressed(KeyCode::ArrowUp) {
        *human_rotate_instr = Some(RotateInstr::Clockwise);
    }
    if keyboard.just_pressed(KeyCode::KeyQ) {
        *human_rotate_instr = Some(RotateInstr::Counterwise);
    }

    for (mut game, controller) in games.iter_mut() {
        let mut move_instr = None;
        let mut rotate_instr = None;
        match controller.into_inner() {
            Controller::Human => {
                move_instr = *human_move_instr;
                rotate_instr = *human_rotate_instr;
            }
            Controller::AI(network) => {
                let mut inputs = vec![-1.0; game.width * game.height];
                for y in 0..game.height {
                    for x in 0..game.width {
                        if game.get(x as isize, y as isize) {
                            inputs[y * game.width + x] = 1.0;
                        }
                    }
                }
                network.update(0.1, &inputs);

                let outputs = network.get_outputs();
                if outputs[0] < -0.9 {
                    move_instr = Some(MoveInstr::GoLeft)
                }
                if outputs[0] > 0.9 {
                    move_instr = Some(MoveInstr::GoRight)
                }

                if outputs[1] < -0.9 {
                    rotate_instr = Some(RotateInstr::Counterwise)
                }

                if outputs[1] > 0.9 {
                    rotate_instr = Some(RotateInstr::Clockwise)
                }
            }
        }

        if *ticks % 7 == 0 {
            game.tick_input(move_instr, rotate_instr);
            *human_move_instr = None;
            *human_rotate_instr = None;
        }

        if *ticks % 14 == 0 {
            game.tick_gravity();
        }
    }
}
