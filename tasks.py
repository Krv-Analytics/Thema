# File: tasks.py
# Last Update: 04-20-24
# Updated By: SW


from invoke import task, Collection


#                   ╭─────────────────────────────────────╮
#                   │      Generator Commands             |
#                   ╰─────────────────────────────────────╯


@task
def m(c):
    """Creates the multiverse"""
    c.run("python scripts/createInnerSystem.py")
    c.run("python scripts/createOuterSystem.py")
    c.run("python scripts/createUniverse.py")


@task
def i(c):
    """Creates the inner solar system"""

    c.run("python scripts/createInnerSystem.py")


@task
def o(c):
    """Creates the outer solar system."""
    c.run("python scripts/createOuterSystem.py")


@task
def u(c):
    """Creates the universe"""
    c.run("python scripts/createUniverse.py")


#                   ╭─────────────────────────────────────╮
#                   │      Cleaning Script Commands       |
#                   ╰─────────────────────────────────────╯


@task
def clean(c):
    """Destroys the multiverse"""
    c.run("python scripts/sweeper.py")


@task
def cleani(c):
    """Destroys the inner solar system"""
    c.run("python scripts/sweeper.py inner")


@task
def cleano(c):
    """Destroys the outer solar system"""
    c.run("python scripts/sweeper.py outer")


@task
def cleanu(c):
    """Destroys the universe"""
    c.run("python scripts/sweeper.py universe")


#                   ╭─────────────────────────────────────╮
#                   │      Setup Script Commands          |
#                   ╰─────────────────────────────────────╯


@task
def condaenv(c):
    """Generates this project's conda environment"""
    c.run("python scripts/setup.py create-env")


@task
def rmcondaenv(c):
    """Removes this project's conda environment"""
    c.run("python scripts/setup.py remove-env")


#                   ╭─────────────────────────────────────╮
#                   │           Help Command              |
#                   ╰─────────────────────────────────────╯


@task
def help(c):
    """
    Displays usage information for all tasks.
    """
    print("\n\n ---------------------------------------")
    print("     Welcome to Scripting in THEMA   ")
    print("     A product by Krv Analytics  ")
    print(" ---------------------------------------")
    print("\nOptions:")
    tasks = [
        value
        for name, value in globals().items()
        if callable(value)
        and getattr(value, "__module__", None) == __name__
        and value.__name__ != "help"
    ]
    for task in tasks:
        print(f"\nCommand: {task.name}")
        print(f"    Description: {task.__doc__}")
    print("\n\n --------------------------------------- ")
    print("     Go and create something great!  ")
    print(" --------------------------------------- \n\n")


# Create a namespace collection and add tasks
ns = Collection()
ns.add_task(m)
ns.add_task(i)
ns.add_task(o)
ns.add_task(u)
ns.add_task(clean)
ns.add_task(cleani)
ns.add_task(cleano)
ns.add_task(cleanu)
ns.add_task(condaenv)
ns.add_task(rmcondaenv)

# Set the default task to 'help'
ns.add_task(help, default=True)
