#[macro_use]
extern crate log;
extern crate env_logger;
extern crate leaf;
extern crate collenchyma as co;
extern crate collenchyma_nn as conn;

#[cfg(test)]
mod network_spec {
    use std::rc::Rc;
    use std::sync::{Arc, RwLock};
    use co::backend::{Backend, BackendConfig};
    use co::tensor::*;
    use co::framework::IFramework;
    use co::frameworks::Native;
    use co::frameworks::Cuda;
    use conn::plugin::*;
    use leaf::network::*;
    use leaf::layer::{LayerConfig, LayerType};
    use leaf::shared_memory::Blob;
    use env_logger;

    fn backend() -> Rc<Backend<Cuda>> {
        let framework = Cuda::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        Rc::new(Backend::new(backend_config).unwrap())
    }

    #[test]
    fn new_network() {
        let cfg = NetworkConfig::default();
        Network::from_config(backend(), &cfg);
    }

    #[test]
    fn single_sigmoid_layer_forward() {
        let _ = env_logger::init();
        let mut cfg = NetworkConfig::default();
        // set up input
        cfg.inputs.push("in".to_owned());
        cfg.input_shapes.push(vec![1, 30, 30]);
        cfg.inputs.push("label".to_owned());
        cfg.input_shapes.push(vec![1, 1, 10]);
        // set up sigmoid
        let mut sig_cfg = LayerConfig::new("sig".to_owned(), LayerType::Sigmoid);
        sig_cfg.bottoms.push("in".to_owned());
        sig_cfg.tops.push("sig_out".to_owned());
        cfg.layers.push(sig_cfg);
        // set up softmax_loss
        let mut loss_cfg = LayerConfig::new("loss".to_owned(), LayerType::SoftmaxLoss);
        loss_cfg.bottoms.push("sig_out".to_owned());
        loss_cfg.bottoms.push("label".to_owned());
        cfg.layers.push(loss_cfg);

        let backend = backend();
        let mut network = Network::from_config(backend.clone(), &cfg);
        let loss = &mut 0f32;
        let inp = Blob::from_data(SharedTensor::<f32>::new(backend.device(), &vec![1, 30, 30]).unwrap());

        let inp_lock = Arc::new(RwLock::new(inp));
        network.forward(&[inp_lock], loss);
        println!("LOSS: {:?}", loss);
    }
}
