---
title: "WebAssembly"
date: 2019-12-18T15:29:39+08:00
showDate: true
draft: false
---

# WHAT

> WASM is a binary instruction format for a stack-based virtual machine. It's designed as a portable target for compilation of high-level languages like C++, enabling deployment on the web for client and server applications. It defines an Abstract Syntax Tree (AST) which gets stored in a **binary format**

Brief:

- JS is a standard with doc of per line behaviour
- Browser uses a JS engine (aka Virtual Machine) to **interpret** and **run** JS code based on the official specs
- In short, WASM is a W3C standard defining a binary format with **.wasm** extension file containing machine instructions (i.e. compiled code)
- Any engine can run `.wasm` files if standard implemented
- KEY: WASM is intended to run on web browser's JS engines! All major browsers added support so can call wasm code from JS and vice-versa

How it works:

1. Dev code in C++
2. Compile with WA support into WA **byte-code** files
3. Browser JS Engine translates WASM into machine code and runs it

![Logic Flow](https://cdn-images-1.medium.com/max/1600/0*xU7akQpF9KctXbQA.png)

DEMO - Counting

1. Create a couple of C funcs
2. Compile C code to wasm file; multiple ways for this process
3. Web application JS can fetch wasm file and call C funcs (which is now in wasm binary format)
4. Lastly, user click triggers JS calling the C func and uses the output to update DOM

```bash
# file dir
node_modules
index.c
index.css
index.html
index.js
index.wasm
```

```html
<body>
    <h1>
        Web Assembly Demo
    </h1>
    <button id="btn-doit">
        Do it!
    </button>
    <div class="display-area" id="display-area-doit">
        <p>
            0
        </p>
    </div>
    
    <button id="btn-count">
        Count
    </button>
    <div class="display-area", id="display-area-count">
        <p>
            0
        </p>
    </div>
    
    <script src="index.js"></script>
</body>
```

```c
// index.c
static int count = 0;

int CCount(int increaseValue) {
    count = increaseValue + count;
    return count;
}

int CDoit() {
    return 42;
}
```

```javascript
fetch('index.wasm').then(response =>
                        response.arrayBuffer()
                        ).then(bytes =>
                              WebAssembly.instantiate(bytes, importObject)
                              ).then(results => {
    // Get access to WASM code
    const wasmExports = results.instance.exports;
    
    doItButton.onclick = ()=> {
        doItDisplayArea.innerText = wasmExports.CCount();
    }
    
    counterButton.onclick = ()=> {
        counterDisplayArea.innerText = wasmExports.CDoit(1);
    }
```



**WHY**

- HPC web apps - CPU-intensive on **front end**
- Native interop with JS sharing objects two-way
- Use case:
  - game
  - ML
  - VR
  - Speed streaming editing
  - Image recognition and visualisation and simulation etc



## WASM-RUST Web App

![Flow](https://miro.medium.com/max/875/1*jxw6m_ObbwHuRFhmYpsBbg.png)

> Live update possible (Rust, C++) have live-reloading and hot-module-replacement auto-display changes

Support means:

- Production-grade compilers and tool-chains generating WASM
- Active community and helper tools
- Less experimental libraries

**Mainly due to WASM supports ONLY flat linear memoary - good for C++/Rust but other needs GC to run**

Demo - updating click with Rust-Wasm func rather than JS 

- overkill sure, more for say hashing function or etc

- Requisites:
  - NodeJS (https://nodejs.org/en/download/)
  - Rust toolchain (https://www.rust-lang.org/tools/install)
  - Wasm-Pack (https://rustwasm.github.io/wasm-pack/installer/)

Tutorial (https://medium.com/tech-lah/webassembly-part-ii-a-wasm-with-rust-2356dbc6526e)

