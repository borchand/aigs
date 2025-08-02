#import "@local/lilka:0.0.0": *
#import "@preview/lovelace:0.3.0": *
#import "@preview/wrap-it:0.1.1": wrap-content
#show: lilka

#set text(size: 12pt)
#set par(justify: true, leading: 1.2em)

#align(center, text(1.5em, tracking: 0.1em, upper[Quality Diversity]))
#v(2em)

Quality diversity algorithms, rather than finding a specific good solution to a
problem, construct archives of different high performing solutions. "Different"
can be defined in various ways, an obvious example being "solutions that look
different to a vision model" @kumarAutomatingSearchArtificial2024. A
popular#footnote[At least in this lab] quality diversity algorithm is the nicely
named Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)
@mouretIlluminatingSearchSpaces2015.


#[
  #show heading.where(level: 1): set heading(numbering: none)
  = Index of Sources <touying:unoutlined>
  #bibliography("zotero.bib", title: none, style: "ieee")
]
