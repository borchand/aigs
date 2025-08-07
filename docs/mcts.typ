#import "@local/lilka:0.0.0": *
#import "@preview/lovelace:0.3.0": *
#import "@preview/wrap-it:0.1.1": wrap-content
#show: lilka

#set text(size: 12pt)
#set par(justify: true, leading: 1.2em)

#align(center, text(1.5em, tracking: 0.1em, upper[Monte Carlo Tree Search]))
#v(2em)

This is the first of three labs. Its purpose is to _1)_ allow you to get
hands-on experience with game trees, and _2)_ to ensure that your computer setup
is up and running. Having installed `uv`#footnote[#link(
  "https://docs.astral.sh/uv/",
  "docs.astral.sh/uv",
)], and cloned our repo #footnote[#link(
  "https://github.com/syrkis/aigs",
  "github.com/syrkis/aigs",
)], run `uv sync` to install all dependencies. I recommend using Zed
#footnote[#link(
  "https://zed.dev",
  "zed.dev",
)] or VS Code as your editor, but anything (including Notepad++) will do. We
will explore this lab in the coming sessions.

= Connect four

`aigs/games.py` contains an implementation of Tic-Tac-Toe. You have to:
- Look at the code and understand it
- Create a child class of `Env` called `ConnectFour`
- Implement an `init() -> State` method
- Implement a `step(state, action) -> State` method.

= Minimax

The minimax function has the signature $f : S -> BB -> ZZ$. In plain English: it
takes a state and a boolean (indicating player) and returns an integer
(representing value). When we want to take an action, then, we call minimax for
each of our potential actions, and take the action with the highest value. You
have to:
- Implement the minimax function
- Call it for every potential action at time $t$
- Then take the action that yielded the highest value

= $alpha-beta$ pruning

One common sense modification to minimax is to break early when a particular
branch allows the oponent something better than what we are already guaranteed
to have. The function signature then becomes $f : S -> BB -> ZZ -> ZZ -> ZZ$,
with the second to last two values representing $alpha$ and $beta$ (the values
to beat for the two players). You have to:
- Copy your minimax function
- Add input parameters $alpha$ and $beta$
- Modify the function to break early when appropriate


= Heuristic variations

Notice how we never actually _looked_ at the game board to gaguge its value, but
instead we fully complete games, exhaustively exploring the game tree. Since
already for connect-four, the combinatoric explosion leaves exhaustive search
intractable, Shannon's 1950s chess program used heuristics @shannon1950. Rather
than always fully finishing every simulated game, he'd maximally look $n$ steps
into the future, and then evaluate the board using a _heuristic_—a linear
combination of manually crafted features (e.g.,
$0.3 times "number of pawns in the center" + 0.1 times "mean distance from ally pawn to enemy king"$,
or whatever.) You have to:
- Create a heuristic function that given a board returns a value
- Copy your minimax function, and add a depth parameter
- Modify the function to return winner if terminated or heuristic value of depth
  is 0



= Monte Carlo tree search

Even with $alpha-beta$ pruning, we are still exhaustively searching through the
game tree (in what we know for sure not to be dead ends). Why not just sample?
When faced with a choice of making move $a$ or $b$ we could simulate $n$
potential futures for each possibility and count outcomes. There are many
variations and smart tricks to MCTS @browne2012. You have to
- Implement a rollout function
- Implement a backpropagation function
- Implement a selection function
- Implement an expansion function
- Combine the above into MCTS



#[
  #show heading.where(level: 1): set heading(numbering: none)
  = Index of Sources <touying:unoutlined>
  #bibliography("zotero.bib", title: none, style: "ieee")
]
