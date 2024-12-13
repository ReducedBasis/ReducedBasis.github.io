---
title: 'Home'
date: 2023-10-24
type: landing

design:
  # Default section spacing
  spacing: "6rem"

sections:
  - block: hero
    content:
      title: Documentation for Reduced Basis Methods (RBM)
      text: 
      primary_action:
        text: Read the docs
        url: /docs/
      announcement:
        text: 
        link:
          text: 
          url: 
    design:
      spacing:
        padding: [0, 0, 0, 0]
        margin: [0, 0, 0, 0]
      # For full-screen, add `min-h-screen` below
      css_class: ""
      background:
        color: ""
        image:
          # Add your image background to `assets/media/`.
          filename: ""
          filters:
            brightness: 0.5
  - block: features
    id: features
    content:
      title: Main features
      text: 
       "
       $\\ -$ Pedagogical website on Reduced Basis Methods (non-exhaustive list of methods);
       

$\\ -$ Python notebook with explanations on simple examples (driven cavity, Helmholtz equation, advection-diffusion ...);


$\\ -$ The goal is not to show most efficient implementations but to explain how each method works and which to choose in a given context;"
      items:
        - name: Reduced basis methods
          icon: magnifying-glass
          description: Find the reduced basis method that best matches your problem.
        - name: Documentation
          icon: bolt
          description: Detailed  documentation. 
        - name: Python notebook {{< icon name="python" pack="fab" >}} 
          icon: code-bracket
          description: RBM code notebook with easy and concrete examples.
  - block: cta-card
    content:
      title: "Reduced Basis Methods"
      text: This website provides a basic introduction to Reduced Basis Methods (RBM). They aim at reducing the runtimes of classical methods of resolution (e.g. finite elements method) for parameterized partial differential equations when they have to be solved for many different parameter values. They have many applications arising from engineering and applied sciences, such as real-time simulation or calibration problems. For each RBM, a short description with links to several articles is presented, and a simple application in a Python notebook and links to other computational langages (such as Fenics/Feel++/FreeFem++) are provided.
      button:
        text: Get Started
        url: /docs/
    design:
      card:
        # Card background color (CSS class)
        css_class: "bg-primary-700"
        css_style: ""
---
