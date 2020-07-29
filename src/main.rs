// Uses
use std::path::PathBuf;
fn main() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("shaders");
    path.set_extension("spv");
    println!("p = {:?}", path);
    path.set_extension("comp");
    println!("p = {:?}", path);
}
