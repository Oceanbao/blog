---
title: "GIT"
date: 2019-12-018T15:52:43-05:00
showDate: true
draft: false
---

**ONLY PUT SOURCE FILES INTO VERSION CONTROL, NEVER GENERATED FILES**

- source is any file authoured
- generated file that which computer created

```bash
git init projectName
cd projectName

# credential
git config --global user.name "User"
git config --global user.email "user@user.com"

# staging
git add .

# write to history
git commit -m "message"

# branches [-c:copy, -m:move, -d:delete]
git branch -c my-branch
# checkout to branch
git checkout my-branch

# merge with current branch
git merge my-branch

# remote
git remote add vue https://github.com/vuejs/vue.git

# Changes
# stashes - create tmp records of WIP to return
git stash
git stash list
git stash apply stash@{0}

# cancel all current stage changes
git reset
# revert to commit ID
git revert fc95eb...


```



**Example of gitignore**

```shell
__pycache__
venv
env
.pytest_cache
.coverage
```



**Checkouting paste SHA or label**

```shell
git checkout 946b99.... (20-bit)
Note: checking out....

You are in 'detached HEAD' state. 

If you want to creat ea new branch to retain commits you create, you may do so (now or later) by using -b with the checkout cmd e.g.:

	git checkout -b <new-branch-name>
	
HEAD is now at 956b99b... added myname module
```



- HEAD is Git's name for whatever SHA you happen to be looking at at any time, NOT what is on your filesystem or what is in your staging area - it means what Git thinks you have checked out.
- For example: when looking at (HEAD at) old SHA, HEAD is behind **master** SHA
  - to get back (1) checkout SHA_of_master
  - (2) checkout master

## Commit

`git commit -a` - auto-stage all tracked files (modified tracked files not staged)

`git difftool` - use difftool to diff

`git diff --staged`

`git rm` 

`git rm --cached` - keep file in working tree but remove it from staging 

`git rm log/\*.log` - `\*` due to Git does own filename expansion in addition to shell's filename expansion, removing all `.log` extension in `log/` dir 

`git mv old_name new_name` - combines `mv old new; git rm old; git add new;`

## Log

`git log`

`-p / --patch` 

`--pretty=oneline, short, full, fuller`

`--pretty=format:"<PATTERN>"`

`%H, %h, %T (tree hash), %t, %P, %p, %an, %ae, %ar, %ad, %cn, %ce, %cr, %cd, %s`

`git log -S function_name`

`--since, --after, --until, --before, --author, --committer, --grep`



## Undoing

`git commit --amend` - replace previous commit with new staged 

`git reset HEAD FILE_2_UNSTAGE` - unstage file

`git checkout -- FILE_2_UNMODIFIEd` - revert to original (last commit, initially cloned, or however got into working tree) **this is dangerous - any local changes are gone!**

## Tag

`git tag -a v1.4 -m "detail of `1.4"

`git tag`

`git tag show v1.4`

`git tag -a v1.2 9fceb02` - tag a commit afterwards

`git push origin v1.5` - sharing tag to remote branch

`git push origin --tags` - pushing all tags to remote server 

`git tag -d v1.5` - delete tag

`git push origin :refs/tags/v1.5` - null value before colon is being pushed to remote tag name, effectively deleting it

`git push origin --delete <tagname>` - more intuitively

`git checkout 2.0.0` - view tags

`git checkout -b version2 v2.0.0` - version2 branch will be slightly diff than v2.0.0 tag since it will move forawrd with new changes so do be careful



## Alias

`git config --global alias.co checkout`

`git config --global alias.unstage 'reset HEAD --'`

`git config --global alias.last 'log -1 HEAD'`

## Branching

`git log --oneline --decorate -graph --all` - shows divergence

CASE:

1. do some work on web
2. make branch for new user story
3. do some work in branch
4. critical fix
   1. switch to PROD branch
   2. make branch to add fix
   3. after tested, merge fix and push to PROD
   4. switch back to original user story and resume

- started with single tree of some commits

- start to work on issue #53 

- create new branch 

- `git checkout -b issue53`

- `vim index.html` - do some work moves issue53 forward since checked out (HEAD it)

- `git commit -a -m 'added a new footer [issue 53]`

- apply fix to master - BUT best to have CLEAN working tree when switching branches - (around by stashing and amending) now assuming committed all changes

- `git checkout master`

- `git checkout -b hotfix`  - work

- `vim index.html`

- `git commit -a -m 'fixed the broken email address'`

- `git checkout master` - now merge hotfix back into master to deploy PROD

- `git merge hotfix` - "fast-forward" since hotfix merged in was directly ahead of previous, Git simply moves the pointer forward - in other words, merging one commit with a commit that can be reached by following the first commit's history, Git simplifies things by moving the pointer forward as there is NO divergent work to merge together

- first delete hotfix branch, as no longer need it - the master points at the SAME place

- `git branch -d hotfix`

- `git checkout issue53`

- `vim index.html`

- `git commit -a -m 'finished new footer [issue 53]`

- NOW either `git merge master` to get master into issue53, or wait to integrate changes until decided to pull the issue53 back into master later

- NOW finished, merge issue53 into master, much like before

- `git checkout master`

- `git merge issue53`

- looks different as dev history diverged from some older point, not direct ancestor, Git does a simple three-way merge, using two snapshots pointed to by the branch tips and common ancester of two

- now ok to close ticket of issue53 and delete branch

- CONFLICT (modified the same file on divergent branches)

  - ```html
    <<<<<<<< HEAD: index.html
    <div id="footer">text</div>
    ========
    <div id="footer">
        please content .....
    </div>
    >>>>>>>>>> issue53:index.html
    ```

  - version HEAD (master branch as it's checked out when running merge) is top part, everything above =======, while issue53 branch is below. 

  - can use `git mergetool`

MANAGEMENT

- `git branch -v` - branch with last commit
- `git branch --merged / --no-merged` - which branch is merged into current or not
- `git checkout -b serverfix origin/serverfix` - gives local branch that can work on starts where origin/serverfix is 
- `git checkout --track origin/serverfix` - tracking branch (upstream branch) is local branches that have direct relationship to a remote branch, if tracking and `git pull` Git knows which server to fetch and merge in; clone auto-make master tracking origin/master, BUT can set up other tracking branches if wish - ones tracking branches on other remotes, or not tracking master; e.g. 
- so common that even a shortcut for shortcut - if branch to checkout (a) nonexistent, and (b) exactly matches a name on only one remote, Git creating a tracking branch `git checkout serverfix`
- `git checkout -b sf origin/serverfix` - set up a local branch different name than remote (sf pull from origin/serverfix)
- `git branch -u origin/serverfix` - already a local branch and set it to a remote just pulled down, or change upstream branch tracking
- `git branch -vv` see tracking branches set up and position
- `git pull` - checks what SERVER/BRANCH the CURRENT branch is tracking, fetch + merge IN remote branch
  - better to fetch + merge explicily
- `git push origin --delete serverfix` - deleting remote branch (remove the pointer from the server, server generally lingers data till GC, so easy to recover)

## Rebasing

- another way to integrate changes besides merge
- instead of 3-way merge, apply patch of one on top of master track - take all changes committed on one and replay on another
- `git checkout experiment` + `git rebase master` - works by going to common ancestor, getting diff of each to temp files, resetting current branch to the same commit as branch rebasing onto, and applying each change in turn
- at this point, can go back to master and merge `git checkout master` and `git merge experiment`
- cleaner history!
- Example: mainline (master), branch1 (server) and branch2 (client), all have commits forwards
  - `git rebase --onto master server client` merge client branch into mainline but hold off server, taking client branch, figure out the patches since it diverged from server branch !! and replay them in client as if it was based directly off master instead
  - now can fast-forward master by `git checkout master; git merge client`
  - `git rebase <basebranch> <topicbranch>` pull in server branch `git rebase master server` without having to check it out - replaying server patches on top of master
  - again, fast-forward base master `git checkout master;  git merge server`
  - remove client and server `git branch -d client` and server
- PERILS - **do not rebase commits outside your repo and people may have based work on them**
- 

## Branching

`git checkout -b new_feature`

- Note `log` looks exactly the same when creating new branch, which starts at the location you were at
- Now working on the branch, edit codes -> `git add -A && git commit -m ...`
- Now `log` has new SHA
- Switch back to master `git checkout master`
- Is that new SHA there?
- Git has built-in way to compare state of branches `git show-branch new_feature master`

Example of Comparison:

```shell
# the label repr
* [new_feature] commit 4
! [master] commit 3
--
* [new_feature] commit 4
* [new_feature^] commit 1
* [new_feature~2] added code for featur ex
+ [master] commit 3
+ [master^] commit 2
*+ [new_feature~3] created .gitignore

# SHA repr
git show-branch --sha1-name new_feature master
```



## 3 Ways to Branch In



### Merge

- Git create a new commit combining the top SHAs of two branches if need to.
- If all of commits in other branch are ahead (based on) the top of the current branch, it will just do a fast-forward merge and place those new commits on this branch
- `git checkout master` and `git merge new_feature`
- since merge when on master, the merge of new_feature to master
- if changes made to master before merging, Git would have created a new commit combining both changes
- Git auto-merge based on common ancestors
- BUT if the same section of code has been mod in both branches - Git cannot figure out what to do - **conflict**



### Rebasing

- In merge, if both branches changed, a new **merge commit ** is created
- In rebasing, Git will take the commits from one branch and replay them, one a time, on top of the other branch
- Demo:



### Cherry-Picking

- specify exactly which commits meant
- git check-pick single SHA -> apply a single SHA into current branch



## Remote Repo

clone / fetch / pull / push

clone is simple -> Download

`git remote show origin`

`git fetch <remote>`



### Fetch

- Step back at how Git manages local and remote
- when clone a new repo, Git doesn't just copy down a single version of the files in that project - it copies the entire repo and uses that to create a new repo on local
- Git does not make local branches for you except for master - BUT it does keep track of the branches on the server - Git creating a set of branches all start with `remotes/origin/<branch_name>`
- Only rarely (almost never), will one check out these branches, but it's handy to know that they are there
- when creating a new branch and the name matches an existing branch on the server, Git will mark local branch as `tracking branch` associated with a remote
- All `git fetch` does is **update all of the remotes/origin branches** mod only the branches stored in remotes/origin and not any of local branches



### Pull

- simply a combo of the above two - first fetch update all branches, then if the branch you ar eon is tracking a remote branch, then it does a git merge of the corresponding remote/origin branch to your branche
- For example: you were on new_feature, coworker just added some code to it on server; if `git pull`, Git will update ALL of the 'remotes/origin' branches and then do a `git merge remotes/origin/new_feature` which will get the new commit onto the local branch HEAD
- LIMIT: Git won't let you even try to do `git pull` if having modified files on local - that can create too much of a mess
- If local and remote **diverged**, `git merge` portion of `pull` will create a merge commit - or rebase by `git pull -r`



## Simple Git Workflow

1. `git status` make sure current area is clean
2. `git pull` get latest version from remote - saving merging issues later
3. Edit files and make changes - test and linter
4. `git status` find all files changed - make sure watch untracked files too
5. `git add [files]` add the changed files to the staging area
6. `git commit -m "message"` make new commit
7. `git push origin [branch-name]` push changes up remote



## Pro Git

https://realpython.com/advanced-git-for-pythonistas/

