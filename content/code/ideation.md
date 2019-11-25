---
title: "Ideation"
date: 2019-10-04T17:42:33+08:00
showDate: true
draft: false
toc: true
---

# Steve Klabnik: WASM + Rust + Serverless

## Rust

- Fast and Memory safe language 
- Hard to learn but better programming
- Close tie with WebAssembly

## WASM

- Originated as asm.js - a fast subset of JS
- Developped into an universal binary runnable on web browser and elsewhere
- WASI the runtime interface for other languages (see Lin Clark's talk)

## Serverless

- With Rust and WASM, any app can run anywhere

# Rasa NLU Lessons

1. Train INTENT and MESSAGE in single NN to be language agnostic (any lang tokenizable)
2. Out-of-data and Fallback policies to improve service
3. Form collection required from user before Decision
4. Feedback loop of system itself logging to improve the system (i.e. logging user-bot conversation to extract more labels)

# spaCy Conference Talks

## Mark Neumann: SciSpacy and Non-ML Extension

### Build Domain Model

1. Entities
2. POS + Dependencies
3. Vector
4. Mixing in generic text to have both worlds

[spaCy IRL slide](https://docs.google.com/presentation/d/1CEc3pTMLX-XV1zgirydhURZdcNlgB2l4TXQv6mxLhy8/edit#slide=id.g5afab164be_0_116)

## Quartz Language Model

[link to slide](https://speakerdeck.com/ddqz/a-natural-language-pipeline?slide=27)

