# 20230208 Teris and Julien chat - Development coordination

## Agenda proposal by J

### Objectives and current state of affairs
- should we aim for some kind of a fork of your previous work?
  - what do we think we should definitely adopt, e.g. xarray
  - what do you think was not great or should be designed differently from what you've done previously
  - what really differs enough between the two project that will involve significant design differences
- what to recover from current pipelines?
- what needs to be done to have them exploitable in a new framework?
- **how can we maintain-improve short/medium-term usability, for observability of current cohorts, and presentation at ASAP poster (2nd of March). This is kind of crucial for everyone short-term I believe.**

### Organisation:

- How can we divide the work in several components / phases
- What is the most logical order to implement that
- How do we organize to not be redundant / work efficiently together

## Minutes

### From previous Teris work
- Regarding his previous pipelines, Teris wants to move away as far as possible from too much object-oriented programming relying excessively on object states. Instead, we should aim at storing simple file formats. E.g. separate dataframes for data and metadata or even config files, later xarray, but not huge objects. (For the sake of avoiding too much dependencies between order of execution of the methods, and also to be able to unpickle files if we modified a class/method)
- J & T agreed that xarrays and snake files are a good practice medium-to-long term.

### Organisation of the work together
- J & T agreed that the main repository will remain trialexp on Julien github, at the very least until we'll do a major upgrade.
- T showed J a bunch of tips about PR (Pull Requests) comments and review (on the github website interface)
- So T is working on a fork under his github and will regularly od PR and/or merge himself.
  
### Immediate objectives
- J sharing with T objectives for ASAP poster
  - interests and evolution of the task
  - behavioural results of previous and current cohort
  - photometry examples from 316 and most likely from ongoing cohort
  - simple illustrative examples from Neuropixels data
- **The importance of being able to make progress on the pipelines while maintaining the abilities of experimenters to take a quick look at the data was shared. Also important for ASAP poster**

### Solutions and work to do
  - A convenient way to do that is possibly to implement new functions as standalalone methods while keeping (past) object-methods functional. This will also likely facilitate the transition to smaller file and less object-state-dependant programming.
  - T will work more closely on Behaviour
  - J will work more closely on Photometry
  - J & T will coordinate to be as consistent as possible, and T to commit "scaffolding" folders and files for J to imitate the template
  - Several discussion on the implementation of event / trial data, and ways to store metadate. 
    - This reach the point where current solution of storing (behavioural) data on dataframe, and metadata on a separate one with a similar index will probably keep being a good solution at least medium term.
    - J to provide T with an example of such data and metadata DataFrames grouping several animals/sessions/conditions