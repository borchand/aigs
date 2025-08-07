#import "@local/lilka:0.0.0": *
#import "@preview/lovelace:0.3.0": *
#import "@preview/wrap-it:0.1.1": wrap-content
#show: lilka

#set text(size: 12pt)
#set par(justify: true, leading: 1.2em)

#align(center, text(1.5em, tracking: 0.1em, upper[Large Language Models]))
#v(2em)

A language model is a function that assigns probability to a sequence of text.
$square.filled$


#[
  #show heading.where(level: 1): set heading(numbering: none)
  = Index of Sources <touying:unoutlined>
  #bibliography("zotero.bib", title: none, style: "ieee")
]
