use glsl_to_spirv::{compile, ShaderType};
use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};
pub fn main() {
    for s_name in ["sgemm1", "sgemm6"].iter() {
        let mut shader_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        shader_path.push("shaders");
        let mut shader_path = shader_path.join(s_name);
        shader_path.set_extension("comp");
        let mut file = File::open(&shader_path)
            .expect(format!("File path {:?} does not exist", &shader_path).as_str());
        let mut buf = String::new();
        file.read_to_string(&mut buf).unwrap();
        let mut cs = compile(buf.as_str(), ShaderType::Compute).unwrap();
        shader_path.set_extension("spv");
        let mut buf = Vec::new();
        cs.read_to_end(&mut buf).unwrap();
        let mut file = File::create(&shader_path).unwrap();
        file.write_all(&buf).unwrap();
    }
}
