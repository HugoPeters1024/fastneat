//! Renders a 2D scene containing a single, moving sprite.
use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
};
use bevy_editor_pls::prelude::*;

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
    x: usize,
    y: usize,
}

#[derive(Component)]
struct Game {
    board: Vec<bool>,
    width: usize,
    height: usize,
    current_piece: Piece,
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
            },
        }
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        self.board[self.width * y + x]
    }

    pub fn set(&mut self, x: usize, y: usize) {
        self.board[self.width * y + x] = true;
    }

    pub fn tick(&mut self) {
        let mut conflict = false;
        'outer: for dy in 0..3 {
            for dx in 0..3 {
                let x = self.current_piece.x + dx;
                let y = self.current_piece.y + dy + 1;
                if PIECES[self.current_piece.kind][dy * 3 + dx] == '#' && self.get(x, y) {
                    conflict = true;
                    break 'outer;
                }
            }
        }

        if conflict {
            for dy in 0..3 {
                for dx in 0..3 {
                    let x = self.current_piece.x + dx;
                    let y = self.current_piece.y + dy;
                    if PIECES[self.current_piece.kind][dy * 3 + dx] == '#' {
                        self.set(x, y);
                    }
                }
            }

            self.current_piece.x = 0;
            self.current_piece.y = 0;
            self.current_piece.kind += 1;
            self.current_piece.kind %= PIECES.len();
        } else {
            self.current_piece.y += 1;
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

const PIECES: [[char; 9]; 7] = [
    ['.', '#', '.', '.', '#', '.', '.', '#', '.'], // i block
    ['#', '#', '.', '#', '#', '.', '.', '.', '.'], // o block
    ['.', '#', '.', '#', '#', '#', '.', '.', '.'], // t block
    ['.', '#', '.', '.', '#', '.', '.', '#', '#'], // l block
    ['.', '#', '.', '.', '#', '.', '#', '#', '.'], // j block
    ['.', '#', '#', '#', '#', '.', '.', '.', '.'], // s block
    ['#', '#', '.', '.', '#', '#', '.', '.', '.'], // z block
];

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
    let mut game = Game::new(width, height);

    for i in 0..10 {
        game.set(9 - i, 9 + i);
    }

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
            for y in 0..game.height {
                for x in 0..game.width {
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

    let game = commands.spawn(game).id();
    render.game_to_render = Some(game);

    commands.insert_resource(all_assets);
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
            if game.get(x, y) {
                *visibility.get_mut(block_entity).unwrap() = Visibility::Visible
            }
        }
    }

    for dx in 0..3 {
        for dy in 0..3 {
            let x = game.current_piece.x + dx;
            let y = game.current_piece.y + dy;
            let block_entity = render.board[y * game.width + x];
            if PIECES[game.current_piece.kind][dy * 3 + dx] == '#' {
                *visibility.get_mut(block_entity).unwrap() = Visibility::Visible
            }
        }
    }
}

fn tick_games(mut games: Query<&mut Game>, mut ticks: Local<usize>, keyboard: Res<ButtonInput<KeyCode>>) {
    *ticks += 1;
    if *ticks < 38 {
        return;
    }
    *ticks = 0;

    for mut game in games.iter_mut() {
        if keyboard.pressed(KeyCode::ArrowRight) {
            game.current_piece.x += 1;
        }
        if keyboard.pressed(KeyCode::ArrowLeft) {
            game.current_piece.x -= 1;
        }
        game.tick();
    }
}
