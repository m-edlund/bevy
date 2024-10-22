#!/bin/bash


# Add the wasm target if not already added
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen-cli if not already installed
cargo install -f wasm-bindgen-cli

# Build the example for wasm
cargo build --example readback_wasm --target wasm32-unknown-unknown --release

# Generate JS bindings with wasm-bindgen
wasm-bindgen target/wasm32-unknown-unknown/release/examples/readback_wasm.wasm --out-dir wasm_example --target web 

# Create index.html if it doesn't exist
if [ ! -f wasm_example/index.html ]; then
    cat <<EOT >> wasm_example/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Bevy Wasm Example</title>
    <style>
        body, html {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        canvas {
            display: block;
        }
    </style>
</head>
<body>
    <script type="module">
        import init from './readback_wasm.js';
        init();
    </script>
</body>
</html>
EOT
fi

# Start web server
basic-http-server wasm_example
