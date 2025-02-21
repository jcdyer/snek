use std::{
    io::{Read, Write},
    os::fd::AsRawFd,
    process::ExitCode,
    thread::sleep,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use termios::{TCSAFLUSH, Termios, tcsetattr};

const WIDTH: u8 = 120;
const HEIGHT: u8 = 50;
const SECOND: Duration = Duration::from_secs(1);
const GROWTH_FACTOR: usize = 5;

#[derive(Copy, Clone, Debug)]
enum Dir {
    N,
    E,
    S,
    W,
}

macro_rules! die {
    ($exp:expr) => {{
        match $exp {
            Ok(__val) => __val,
            Err(err) => {
                eprintln!("error: {err}");
                ::std::process::exit(1);
            }
        }
    }};
}

fn enable_raw_mode() -> Result<(), std::io::Error> {
    use termios::{
        BRKINT, CS8, ECHO, ICANON, ICRNL, IEXTEN, INPCK, ISIG, ISTRIP, IXON, OPOST, TCSAFLUSH,
        VMIN, VTIME, tcsetattr,
    };
    let stdin = std::io::stdin().as_raw_fd();
    let mut termios = die!(Termios::from_fd(stdin));
    termios.c_cflag &= !(CS8);
    termios.c_iflag &= !(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    termios.c_lflag &= !(ECHO | ICANON | IEXTEN | ISIG);
    termios.c_oflag &= !(OPOST);
    termios.c_cc[VMIN] = 0;
    termios.c_cc[VTIME] = 0;
    tcsetattr(stdin, TCSAFLUSH, &termios)?;
    Ok(())
}

#[derive(Debug)]
struct TerminalGuard {
    term_in: Termios,
    term_out: Termios,
}

impl TerminalGuard {
    fn new() -> Result<Self, std::io::Error> {
        let term_in = Termios::from_fd(std::io::stdin().as_raw_fd())?;
        let term_out = Termios::from_fd(std::io::stdout().as_raw_fd())?;
        enable_raw_mode()?;
        Ok(Self { term_in, term_out })
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        print!("\x1b[?1049l");

        let stdin = std::io::stdin().as_raw_fd();
        let stdout = std::io::stdout().as_raw_fd();
        die!(tcsetattr(stdin, TCSAFLUSH, &self.term_in));
        die!(tcsetattr(stdout, TCSAFLUSH, &self.term_out));
    }
}

enum Input {
    Up,
    Down,
    Left,
    Right,
    Quit,
}

static KEYCODE_UP: &[u8] = b"\x1b[A";
static KEYCODE_DOWN: &[u8] = b"\x1b[B";
static KEYCODE_LEFT: &[u8] = b"\x1b[D";
static KEYCODE_RIGHT: &[u8] = b"\x1b[C";
static KEYCODE_W: &[u8] = b"w";
static KEYCODE_S: &[u8] = b"s";
static KEYCODE_A: &[u8] = b"a";
static KEYCODE_D: &[u8] = b"d";

impl Input {
    /// Parse an input off the front.  Returns the input and the number of bytes consumed
    fn from_slice(slice: &[u8]) -> (Option<Input>, usize) {
        if slice.is_empty() {
            (None, 0)
        } else if slice.starts_with(b"q") || slice.starts_with(b"Q") {
            (Some(Input::Quit), 1)
        } else if slice.starts_with(KEYCODE_UP) {
            (Some(Input::Up), KEYCODE_UP.len())
        } else if slice.starts_with(KEYCODE_DOWN) {
            (Some(Input::Down), KEYCODE_DOWN.len())
        } else if slice.starts_with(KEYCODE_LEFT) {
            (Some(Input::Left), KEYCODE_LEFT.len())
        } else if slice.starts_with(KEYCODE_RIGHT) {
            (Some(Input::Right), KEYCODE_RIGHT.len())
        } else if slice.starts_with(KEYCODE_W) {
            (Some(Input::Up), KEYCODE_W.len())
        } else if slice.starts_with(KEYCODE_S) {
            (Some(Input::Down), KEYCODE_S.len())
        } else if slice.starts_with(KEYCODE_A) {
            (Some(Input::Left), KEYCODE_A.len())
        } else if slice.starts_with(KEYCODE_D) {
            (Some(Input::Right), KEYCODE_D.len())
        } else if slice.starts_with(b"\x1b") {
            // Escape code await further instructions, maybe?
            (None, 1)
        } else {
            (None, 1)
        }
    }
}
fn main() -> ExitCode {
    let _term_guard = die!(TerminalGuard::new());
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();

    let width: u8 = WIDTH;
    let height: u8 = HEIGHT;
    let mut snake_buffer = [(u8::MAX, u8::MAX); 1024];
    let mut snake_len: usize = 11;
    let mut apple = None;
    let mut apple_ticks = width as u16 + height as u16;
    let mut apple_ticks_remaining = apple_ticks;
    let mut snake_head: usize = snake_len - 1;
    let mut snake_tail: usize = 0;
    let mut dir = Dir::N;
    let mut score = 0;

    for (i, segment) in snake_buffer[..snake_len].iter_mut().enumerate() {
        *segment = (width / 2, (height / 2) + snake_len as u8 - i as u8)
    }
    let mut loc = snake_buffer[snake_head];

    let mut timer = Timer::start(SECOND.mul_f32(0.2));
    let mut rng = random::Rng::new(
        std::num::NonZero::new(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u32, // truncate
        )
        .unwrap_or(std::num::NonZero::<u32>::MAX),
    );

    // Switch to alternate buffer
    print!("\x1b[?1049h");
    // Hide the cursor
    print!("\x1b[?25l");
    let mut input_buf = [0; 16];
    let mut input_buflen = 0;
    loop {
        timer.tick();
        if apple.is_none() || apple_ticks_remaining == 0 {
            apple_ticks_remaining = apple_ticks;
            let random = rng.random();
            let random = random % ((width - 2) as u32 * (height - 2) as u32);

            apple = Some((
                (random % (width as u32 - 2) + 1) as u8,
                (random / (width as u32 - 2) + 1) as u8,
            ));
        }
        apple_ticks_remaining -= 1;

        let ct = die!(stdin.read(&mut input_buf[input_buflen..]).or_else(|err| {
            if let std::io::ErrorKind::WouldBlock = err.kind() {
                Ok(0)
            } else {
                Err(err)
            }
        }));
        input_buflen += ct;

        let mut buf = input_buf[..ct].as_mut();
        while !buf.is_empty() {
            let (input, ct) = Input::from_slice(buf);
            buf = &mut buf[ct..];
            input_buflen -= ct;
            dir = match input {
                Some(Input::Up) => Dir::N,
                Some(Input::Down) => Dir::S,
                Some(Input::Left) => Dir::W,
                Some(Input::Right) => Dir::E,
                Some(Input::Quit) => return ExitCode::SUCCESS,
                None => dir,
            }
        }

        snake_head += 1;

        if snake_head == snake_buffer.len() {
            snake_head = 0;
        }
        if snake_tail == snake_buffer.len() {
            snake_tail = 0;
        }
        loc = match dir {
            Dir::N => (loc.0, loc.1 - 1),
            Dir::E => (loc.0 + 1, loc.1),
            Dir::S => (loc.0, loc.1 + 1),
            Dir::W => (loc.0 - 1, loc.1),
        };
        snake_buffer[snake_head] = loc;

        let effective_head =
            snake_head + (snake_buffer.len() * ((snake_head <= snake_tail) as usize));

        if effective_head - snake_tail >= snake_len {
            snake_tail += 1;
            if snake_tail == snake_buffer.len() {
                snake_tail = 0;
            }
        }

        if Some(snake_buffer[snake_head]) == apple {
            snake_len += GROWTH_FACTOR;
            apple = None;
            timer.interval = timer.interval.mul_f32(0.95);
            timer.start = Instant::now();
            timer.tick = 0;

            apple_ticks = ((apple_ticks * 19) / 20).max(WIDTH.min(HEIGHT) as u16);
            score += 1;
        }

        let collide = if loc.0 >= width || loc.0 == 0 || loc.1 >= height || loc.1 == 0 {
            true
        } else if snake_head > snake_tail {
            snake_buffer[snake_tail..snake_head].contains(&loc)
        } else {
            snake_buffer[..snake_head].contains(&loc) || snake_buffer[snake_tail..].contains(&loc)
        };

        render_field(
            width,
            height,
            &snake_buffer,
            snake_head,
            snake_tail,
            apple,
            score,
            collide,
        );

        if collide {
            sleep(SECOND.mul_f32(1.5));
            break;
        }
    }
    ExitCode::SUCCESS
}

#[allow(unused_variables, clippy::too_many_arguments)]
fn render_field(
    width: u8,
    height: u8,
    snake_buffer: &[(u8, u8)],
    snake_head: usize,
    snake_tail: usize,
    apple: Option<(u8, u8)>,
    score: u8,
    collide: bool,
) {
    // Erase the screen
    print!("\x1b[2J");
    // Jump to top left
    print!("\x1b[H");

    let snake_segments = if snake_head <= snake_tail {
        (&snake_buffer[snake_tail..], &snake_buffer[0..snake_head])
    } else {
        (&snake_buffer[snake_tail..snake_head], &[][..])
    };
    let mut buffer_storage = ['\0'; 256];
    let mut write_buffer = [0; 1024];
    for rownum in 0..height {
        // Variable width up to 256 chars
        let row_buffer = &mut buffer_storage[..width as usize];

        // draw the background
        if rownum == 0 {
            for cell in row_buffer.iter_mut() {
                *cell = '━';
            }
            row_buffer[0] = '┍';
            row_buffer[row_buffer.len() - 1] = '┑';
        } else if rownum == 1 {
            for cell in row_buffer.iter_mut() {
                *cell = ' ';
            }
            row_buffer[0] = '│';
            row_buffer[row_buffer.len() - 1] = '│';

            let score = score.to_string().chars().collect::<Vec<char>>();
            let rowlen = row_buffer.len();
            row_buffer[rowlen - 2 - score.len()..][..score.len()].copy_from_slice(&score)
        } else if rownum == height - 1 {
            for cell in row_buffer.iter_mut() {
                *cell = '━';
            }
            row_buffer[0] = '┕';
            row_buffer[row_buffer.len() - 1] = '┙';
        } else {
            for cell in row_buffer.iter_mut() {
                *cell = ' ';
            }
            row_buffer[0] = '│';
            row_buffer[row_buffer.len() - 1] = '│';
        }

        // draw the apple
        if let Some(apple) = apple {
            if apple.1 == rownum {
                row_buffer[apple.0 as usize] = '♨'
            }
        }
        // draw the snake
        for segment in snake_segments.0.iter().chain(snake_segments.1.iter()) {
            if segment.1 == rownum {
                row_buffer[segment.0 as usize] = 'O';
            }
        }
        if snake_buffer[snake_tail].1 == rownum {
            row_buffer[snake_buffer[snake_tail].0 as usize] = 'o';
        }
        if snake_buffer[snake_head].1 == rownum {
            row_buffer[snake_buffer[snake_head].0 as usize] = '@';
        }
        let write_buf =
            write_to_buf(&mut write_buffer, &*row_buffer).expect("buffer is large enough");
        print!("{}\r\n", write_buf);
    }

    // Check for collisions
    if collide {
        // Jump to row, col
        print!(
            "\x1b[{row};{col}H",
            row = height / 7 * 3, // Just above center
            col = width / 2 - 4
        );
        // Render the boom in a box
        print!(
            "          {nl} ******** {nl} * boom * {nl} ******** {nl}          {nl}",
            nl = "\n\x08\x08\x08\x08\x08\x08\x08\x08\x08\x08",
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Utf8WriteError;

fn write_to_buf<'a>(dest: &'a mut [u8], src: &[char]) -> Result<&'a str, Utf8WriteError> {
    let mut idx = 0;
    let expected_len = src.iter().map(|c| c.len_utf8()).sum();
    if dest.len() < expected_len {
        return Err(Utf8WriteError);
    }
    for ch in src {
        idx += ch.encode_utf8(&mut dest[idx..]).len();
    }
    Ok(std::str::from_utf8(&dest[..idx])
        .expect("we just encoded this from chars so it should be valid"))
}

struct Timer {
    start: Instant,
    interval: Duration,
    tick: u32,
}

impl Timer {
    fn start(interval: std::time::Duration) -> Timer {
        Timer {
            start: Instant::now(),
            interval,
            tick: 0,
        }
    }

    fn tick(&mut self) {
        while self.start.elapsed() < self.interval * self.tick {
            sleep(Duration::from_millis(5));
        }
        self.tick += 1;
    }
}

#[cfg(test)]
mod tests {
    use crate::{Utf8WriteError, write_to_buf};

    #[test]
    fn write_buf_boundaries() {
        let mut dest = [0; 12];

        let src = [];
        let out = write_to_buf(&mut dest, &src).unwrap();
        assert_eq!(out, "");

        let src = ['a', 'b', 'c'];
        let out = write_to_buf(&mut dest, &src).unwrap();
        assert_eq!(out, "abc");

        let src = ['♠', '♥', '♦', '♣'];
        let out = write_to_buf(&mut dest, &src).unwrap();
        assert_eq!(out, "♠♥♦♣");

        let src = ['♠', '♥', '♦', '♣', '!'];
        let err = write_to_buf(&mut dest, &src).unwrap_err();
        assert_eq!(err, Utf8WriteError);
    }
}

/// Introduce randomness using an xorshift-style PRNG.
///
mod random {
    /// Need a random number? Use this.
    pub type Rng = XorShift32;

    /// 32 bit xorshift random number generator
    ///
    /// Described at https://en.wikipedia.org/wiki/Xorshift, based on:
    ///
    /// Marsaglia, G. (2003). ["Xorshift RNGs". *Journal of Statistical
    /// Software*, 8(14), 1–6](https://doi.org/10.18637/jss.v008.i14).
    ///
    /// From the wikipedia article:
    ///
    /// > For execution in software, xorshift generators are among the fastest
    /// > PRNGs, requiring very small code and state. However, they do not pass
    /// > every statistical test without further refinement.
    ///
    /// In other words, it's easy and fast, but crappy.  Perfect.
    pub struct XorShift32 {
        seed: u32,
    }

    impl XorShift32 {
        pub fn new(seed: std::num::NonZero<u32>) -> XorShift32 {
            /* The state must be initialized to non-zero */
            XorShift32 { seed: seed.into() }
        }

        pub fn random(&mut self) -> u32 {
            /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 17;
            self.seed ^= self.seed << 5;
            self.seed
        }
    }
}
