---
title: "VIM"
date: 2019-12-018T15:52:43-05:00
showDate: true
draft: false
---

[toc]

# Modes

1. Normal - `Esc`
2. Insert - `i, a, c`
3. Visual - `v, V, Ctrl-v`
4. Command - `:, /`

## Operators
- `c` - change
- `d` - delete
- `y` - yank into registery
- `~` - swap case
- `gu` - make lower
- `gU` - make uppper
- `!` - filter to external program
- `<` - shift left
- `=` - indent

## Text Objects
- `aw` - a word
- `iw` - inner word
- `ap` - a paragraph
- `at` - a tag
- `it` - inner block

## Motions

- `%` - go to 1st matching paren/bracket
- `[count]+` - down to 1st non-blank char of line
- `[count]$` - to end of line
- `[count]f/F{char}` - to next of {char}
- `[count]h/j/k/l`
- `[count]]m` - go to onset of next method
- `[count]w/W` - go ot word/WORD to the right
- `[count]b/B` - go to word/WORD to the left
- `[count]e/E` - go to end of word/WORD right
- `[count][ops][text obj / motions]`
- `6+` - go down to 6x line
- `gUaW` - cap a WORD
- `3ce` - 3x change to word end
- `4$` - 4x go to end of line
- `d]m` - delete to start of next method

**TRICK**

- 0w - head first word
- d} - enter delete action + motion, e.g. delete till next block
- d% - delete entire content in ()
- C - change rest of cursor
- d/ct + CHAR - delete / change till CHAR
- Ctrl-w + s/v/q - split current window
- :windo - exec for ALL windows
- :sf FILE
- :tabc - close tab
- :tabo - close all other tabs
- :tabf FILE - find and open FILE in new tab
- 
- 

# DEMO

- `vi -U NONE -O *` - open all and splits
- change inside quotation marks `ci'`
- reverse `u`
- find "p" `fp` , delete `x` and paste `p`
- creating index of pwd
  - `:ctags `
  - then go to definition `g<Ctrl-]>`
  - go to file `gf`


## Always be Scrolling

- `zt` - top of page
- `zz` - middle
- `zb` - bottom
- `H, M, L`
- `Ctrl-U/D, Ctrl-F/B, Ctrl-Y/E`

## Basic Editing
- `!:e[dit][++opt][+cmd]{file}`
- `:fin[d][!][++opt][+cmd]{file}`
- `gf`
- `Ctrl-^`

## `/` Search
- `/{patt}[/]<CR>`
- `?{patt}[?]<CR>`
- `[Count]n` - repeat last search [count] times
- `[Count]N` - opposite ibis
- `*` - serach
- `#` - bis opposite
- `gd` - go to local declaration
- `:set hls!` - toggle serach highlightig

## Bookmarks
- `m{a-zA-Z}`
- `:marks` - show all current mraks in use
- ``{mark} , '{mark}` - access
- ``., '.` - jump to last update

## `Ctrl-] and Ctrl-t` - PM

## `Ctrl-O and Ctrl-I` - cycle through `:jumps` and `g;, g` cycle thru `:changes`




### DEMO

**Set bookmarks `'V` as `~/.vimrc` for easy setting**
- after accessing `'V`, jump back `Ctrl-^`
- working dir-tree: `:20vs .` read 20-char visual split current dir
- instead of `:e /path/file` to find and edit, do `:find file.py`
- find file and vertical split `:vert sf FILE`
- change viewpoints `Ctrl-Wx`
- change windows `Ctrl-Ww`

- need to save current buffers viewpoints + windows -> add new tab[windows[buffers]]
  - `:tabf FILE` + `:vert sf FILE` - open tab + find + vert-split new files
- go to next tab - `gt`
- preview files via tree `p` and `Ctrl-Wz` to close
- navigate windows `Ctrl-W hjkl`

- find in current file with {patt} `:vim {patt} %`
  - navigate all matched via quick fix list `:cn<enter>`
  - then do `@:` to repeat {cmd}
- find all matched in all files of interest
  - make arglist first then "grep"
  - `:args **/*.py`
  - `:vim {patt} ##`
- replace {patt} with {patt1}
  - `:cdo s/{patt}/{patt1}/g`
  - navigate back to check `:cp<enter>`

- first, VISUAL mode for drag-selection
  - `Ctrl-V hjkl` (delete `x`)

# Master VIM

**.vimrc**

- test before add to file `:set autoindent` and find doc `:set tabstop?`

```yaml
syntax on 
filetype plugin indent on " Enable file type based on indentation.
set autoindent
set expandtab " Expand tabs to spaces. Essential in Python.
set tabstop=4 " Num. of spaces tab counted for.
set shiftwidth=4 " num. spaces to use for autoindent.
set backspace=2 " fix backspace bevaiour on most terminals.
colorscheme murphy
```

- cycle through built-in colorschemes `:colorscheme + <space> + <Tab>`

**Basic**

- open-edit in vim `:e FILE`
- save-write `:w {FILE} <CR>`  file name optional
- write-quit `:wq` or just quit `:q` or force quit `:q!`

**Swap files**

Default tracking changes in swap files that are auto-gen during editing, recovered via VIM or SSH

- prevent `.swp` littering in current dir set a dir in `.vimrc` by `set directory=$HOME/.vim/swap//`  or disable `set noswapfile`

**Motion**

- `n[hjkl]` n times 
- move to B-next-word `w` , E-closest-word `e`, backward to B-word `b`
- cap above to treat all but white-space as word! **non-space** sequence as WORD

- move paragraph (>2 newlines) `{` and `}`

**Simple edit change**

- `cw`  to change + next word
- `c3e`  (comma counts as a word) 
- `cb` change back word
- `c4l`  intriguing as change 4 chars until `l` 
- NOTE: `cw` == `ce` a legacy
- **structure** `<cmd> <num> <move>/<text-object>`
- example:
  - start by moving to `3<space>` till last WORD `3W`
  - change-delete word + enter into insert mode `cw`
- delete `d` behave as normal with `w` and `e`
- `cc` clears whole line + insert mode + keeping indentation
- `dd` deletes entire line
- tip: `v` enter Visual mode to select before above edits

**Persistent undo and repeat**

- `u` undo last ops and `<Ctrl> r` to redo

- enable persistent undo in vimrc `set undofile`

  - set file fir: 

    ```bash
    if !isdirectory("HOME/.vim/undodir")
    	call mkdir("$HOME/.vom/undodir", "p")
    endif
    set undofir="$HOME/.vim/undodir"
    ```

**loop through keyword in help**

- `:h <keyword> + <Ctrl> + D` to loop through before <CR>
- SEARCH in general: `/search <term>`  - forward and `?search term` backward



## Advanced Editing and Motion

**plugin**

- `mkdir -p ~/.vim/pack/plugins/start`
- vimcr: 
  - `packloadall " Load all plugins`
  - `silent! helptags ALL " Load help files for all plugins`
- install plugin found on GitHub `git clone <.../nerdtree.git> ~/.vim/pack/plugins/start/nerdtree`

**Workspace**

- Buffers = internal repr files for switching between files quickly
- Windows organise workspace by displaying multi-files next to each other
- Tabs a collection of windows
- Folds to hide and expand portions of files, making large files easier to navigate

**Buffers**

- print all by these synonyms `:ls` `:buffers` `:files`
  - `1 %a "file.py" line 30`
    - 1 = buffer number staying constant over Vim session
    - % = current window
    - a = active, loaded, visible
    - line 30 = current cursor pos
- when open multiple files, switching files via buffer number `:b1` or `:b <partial filename> + <Tab>` (for looping)  or next-previous `:bn :bp`
- useful plugin `https://github.com/tpope/vim-unimpaired`

- `:bn` - next buffers
- `:b {filename}` - go to buffer filename
- `:bd` - delete current buffers
- `:buffers` - print all buffers
- `:bufdo {cmd}` - exec cmd to all buffers
- `:n` - next file (based on arglist)
- `:arga {filename}` - add to arglist
- ':argl {files}' - make a local arg copy via files
- `:args` - print all arguments

**arglist**

- `args **/**.yaml` - make arglist with all such files
- followed directly by `:sall` to split all and display in a new tab (`:vert sall` ibis)
- `windo difft` - do for all windows diff on files

**Windows**

- split `:split <file>` or `:sp` or vertical `:vsplit :vs`
- move windows `Ctrl + w [+ hjkl]`

```bash
# binding for ease split in vimrc
" Fast split navigation with Ctrl + hjkl
noremap <c-h> <c-w><c-h>
noremap <c-j> <c-w><c-j>
noremap <c-k> <c-w><c-k>
noremap <c-l> <c-w><c-l>
```

- close `Ctrl + w + q`  or `:bd` delete-buffer-close-window
- close all else `Ctrl + w + o`
- save all and quit `:wqa` !!!
- `:Bd` keeping window on closing buffer `command! Bd :bp | :sp | :bn | :bd`
- movement prefix `<Ctrl> + w`:
  - `H` to leftmost
  - `J` to bottom
  - `K` to top
  - `L` to rightmost
- `Ctrl w r` moves all windows within the row / column to the right or downwards `R` for reverse
- `Ctrl w x` exchanges contents of a window with the next one (or previous one if it's considered a last window)
- `Ctrl w =` equalise h-w of all windows!!!
- resizings:
  - `:resize +N` up height by N rows (`:res`) (or just N)
  - `:vertical resize -N` down width by N columns (`:vert res`)
  - shortcut: `Ctrl w -/+/>/<`

- `:windo {cmd}` - exec for all windows
- `:sf {FILE}` - split windows and `:find {FILE}`
- `:vert {cmd}` - make any cmd use vert split
- `:vnew {FILE}` - open new file in vertical splits

**Tabs**

- used to switch between **collections of windows** for multi-workspace
- often used for different projects or set of files within the same session
- `:tabnew` open new with empty buffer or `+ <file>`
- `gt` to move to next tab, `gT` back
- `:tabclose` or close all windows it contains `:q`
- `tabmove N` to place tab after the Nth tab

**Folds**

- setting auto-folding python files `set foldmethod=indent`
- `zo` open current fold while `zc` close it or `za` to toggle
- visualise pos of folds `:set foldcolumn=N` were N [0, 12]
- `zR` open all `zM` close all

**Folding methods** following `foldmethod`

`indent` - langs having indentation syntax

`expr` - regex based and extremely powerful

`marker` - special markup like `{{{` `}}}`

`syntax` - not every lang supported

`diff` - auto used when operating diff mode

**File tree**

- **Netrw** the built-in file manager 
  - `:Ex` (`:Explore`) or a sym-key `:e .` 
  - `<CR>` opens files/dirs; `-` goes up dir; `D` deletes file/dir; `R` rename
  - `:Vex` opens Netrw vert-split (`:Sex`, `:Lex`)
  - go to remote `:Ex sftp://<domain>/<dir>/` or `:e scp://<domain>/<dir>/<file>`
- **:e with wildmenu enabled** via `set wildmenu`
  - emulating autocomplete `:e + <Tab>` (`Shift + Tab` reverses)
  - `set wildmode=list:longest,full` "complete till longest string then open wildmenu
- **NERDTree** invoking via `:NERDTree`
  - move by `hjkl` or arrow Enter or `o` to open
  - bookmarking while cursor over: `:Bookmark` then `B` to display
  - always display bookmars `let NERDTreeShowBookmarks = 1`
  - toggle `:NERDTreeToggle` or startup enabled `autocmd VimEnter * NERDTree`
  - close it when it's last open window:
    - `autocmd bufenter * if (winnr("$") == 1 && exists ("b:NERDTree") && \ b:NERDTree.isTabTree()) | q | endif`

**plugins**

- Vinegar - `I` toggles help bar and `Shift ~` take to home dir
- CtrlP - fuzzy completion `Ctrl P` shows list of files in project dir 
  - `Ctrl j/k` up/down list of files `f b` to cycle thru options `:CtrlPBuffer` for buffers 

**Movement in Text**

- `t` until followed by character in line while `T` backwards
- `f` find followed by character to search in line `F` backwards
- `_` moves to B of line `$` to end !!!
- word = numbers, letters, underscores / WORD = any but space
- `ge` moves to end of previous word
- `Shift {` to B of paragrah
- `Shift (` to B of sentenes
- `H` to head of window `L` to bottom of window (current)
- `Ctrl f` == page down `Ctrl b` for page up
- `/` followed by string searches document `Shift ?` backwards
- `gg` to top of file
- `G` to end of file

```bash
			gg	
			?
			H
			{
			k
^ F T ( b ge h  l w e ) t f $
			j
			}
			L
			Ctrl+f
			/
			G
```

- move by line numbers enabled by `:set nu`  (put in vimrc for startup)
  - jump `:N` where N is absolute line number
  - jump to N in file `vim file +14`
  - jump `:+N` relative down (`:set relativenumber`)

**Jump into INSERT**

- `a` insert after cursor `A` at end of line (== `$a`)
- `I` insert at B-line after indent
- `o` adds new line below cursor + insert `O` above
- `gi` insert last exit
- `C` deletes txt to the right of cursor til E-line
- `cc` or `S` deletes contents of line 
- `s` deletes single char

**Searching / and ?**

- fastest way to move!!
- Cycling thru matches in the same buffer `n` `N`
- `set hlsearch` (best in vimrc) highlights match (off by `:noh`)
- `set incsearch` makes dynamic move to first match 

**Searching across files**

- `:grep` uses system grep
- `:vimgrep` 
  - `+ <patt> <path>.patt` be string or Vim-like regex 
  - e.g. searching `calc` by `:vimgrep animal **/*.py`
  - navigate `:cn` or `:cp` or visual quickfix window `:copen` then `j k` to jump and enter - close by `:q` or `Ctrl w q` 

**ack**

- spiritual successor of grep for code search `sudo apt install ack-grep`
- `ack --python Animal` 
- Vim plugin integrates `ack.vim` then do `:Ack --python Aniimal`

**Text Object**

- `di)` == delete inside parentheses !!!
- `c2aw` == change outside of two words to delete two words
- `i` inner objects do not include white space or next char while `a` outer do
- common code text objects: `' " ) ] > }`
- Verb `d` or `c` + Number `2` + Adjective `i` or `a` + Noun `)` or `w` etc

**EasyMotion**

- invoke by `\\` + desired movement key

**Registers**

- yank `y` to copy text + movement or text object (usable in visual mode)
- `ye` copy till E-word to copy into register then `p` to paste
- more to see

**Copying from outside VIM**

- `*` register the main clipboard
- `+` register Linux only used for `Ctrl c / v` style
- `"*p` to paste from primary clipboard or `"+yy` to yank a line into Linux's clipboard
- setting default
  - `set clipboard=unnamed " copy into system (*) register`
  - `set clipboard=unnamedplus`



## Plugin

Managers 

- vim-plug - fetch file (`.../master/plug.vim`) then save in `./vim/autoload/plug.vim` (fetch single file `curl -fLo <dest> --create-dirs <source>`)
  - update .vimrc `call plug#begin()` and `call plug#end()`
  - add some plugins between these 2 lines <username>/<repo> format 
    - 
    - 
    - 
    - `Plug 'scrooloose/nerdtree'`
  - save and reload (`:w | source $MYVIMRC`) Do `:PlugInstall` to install
  - `:PlugUpdate` and `:PlugClean` to delete 
  - `:PlugUpgrade` for vim-plug itself + reload file
  - two modes of loading (after above line `, { 'on': 'NERDTreeToggle'}` and `, {'for': 'markdow'}`)
  - For UNIX, add this for transporting to new machine

```bash
if empty (glob ('~/.vim/autoload/plug.vim'))
	silent !curl -fLo ...
	autocmd VimEnter * PlugInstall --sync | source $MYVIMRX
endif
```

Profiling speed

- `vim --startuptime time.log` check bottleneck timestams

**MODE**

- Normal Mode
  - `Ctrl-b/e` B-line and E-line
  - `Ctrl-f` opens editable CLI window with history (as a buffer, edit and exec again, `Ctrl-c` to close)
- Insert mode
  - `Ctrl-o` to do single cmd then back to insert
- Visual and Select mode
  - `v` char-wise `V` line-wise `Ctrl-v` block-wise
  - `o` go to other end of highlighted text to expand-select 
- Replace mode `R` and `r` 
- Terminal mode `:terminal` or `:term` + cmd for single exec

## Text

Autocomplete

- type name of function + `Ctrl n/p` to cycle
- insert-completion mode `Ctrl x` + `Ctrl i` to complete whole line `Ctrl j]` to complete tags `Ctrl f` complets filenames 

YouCompleteMe

- semantic autocomplete; intelligent suggestion; display doc etc
- dependencies `sudo apt install cmake llvm` 
- if using vim-plug: `let g:plug_timeout = 300 \n Plug 'Valloric/YouCompleteMe', { 'do': './install.py'}`
- save files then `:source $MYVIMRC | PlugInstall`
- enable semantic engine `.` in insert or manually `Ctrl space`
- For python jumping to method definition: `noremap <leader>] :YcmCompleter GoTo<cr>` then `\]` when cursor over func call

Navigating code base with tags

- `gd` for local and `gD` for global 

Exuberant Ctags

- External utility generating tag files `sudo apt install ctags`
- `ctags -R .` to navigate project and creating tags file in dir `set tags=tags;` for looking files recursively in parent dir
- `Ctrl ]` + tag to the definition and `Ctrl t` to go back in tag stack
- `:tn` and `:tp` cycle tags `:ts` select manu

## Build, Test, Execute

**Integrating Git Vim - vim-fugitive**

- `:Gstatus` == `git status` but interactive for cycling through files `Ctrl-n/p`
  - `-` stage/unstage file, `cc`or`:Gcommit` , `D` or`:GDiff` open a diff, `Glog` history 
  - Tip: `:copen` pops quickfix window, `:cnext` `:cprevious` navigate
  - `:Gblame`  with shortcuts `C, A, D` resizing, `Enter` opens a diff of chosen commit, `o` opens diff of chosen commit in split, 

**Resovling conflicts with vimdiff**

- in terminal `vimdiff file1 file2`
- move change `]c` forward `[c` backwards
- `do` or `:diffget` moves change to active window
- `dp` or `:diffput` pushes change from active window
- configure Git to use vimdiff as merge tool:
  - `git config --global merge.tool vimdiff`, `git config --global merge.conflictstyle diff3` , `git config --global mergetool.prompt false`
- example: 

```bash
git checkout -b branch-tmp
# edit some codes
git add . && git commit -m "Branch-chnage"
git checkout master
# edit some codes differently
git add . && git commit -m "Master-change"
git merge branch-tmp
# stdout: Conflict
git mergetool 
```

- In view are 4 windows:

  - pos-1 is master branche or local changes
  - pos-2 **closest common ancestor** 
  - pos-3 branch-tmp in conflict
  - pos-4 MERGE shows conflict markers as `<<<< ... >>>>`

  ```bash
  <<<<<< [LOCAL commit/branch]
  [LOCAL change]
  ||||||| merged common ancestors
  [BASE - closest common ancestor]
  =======
  [REMOTE change]
  >>>>>>> [REMOTE commint/branch]
  ```

  > assuming keeping REMOTE (branch-tmp), move cursor to window MERGED (pos-4), move cursor to the next change (`]c` and `[c` to move by changes), and execute `diffget REMOTE`

  - this will change from REMOTE file and place it into MERGED file
    - Get a REMOTE change using `:diffg R`
    - Get a BASE change using `:diffg B`
    - Get a LOCAL change using `:diffg L`
    - Save-exit `:wqa`
    - free to discard `.orig` files after
    - commit