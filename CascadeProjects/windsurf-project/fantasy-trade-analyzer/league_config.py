"""Public league configuration - no sensitive data here."""

# Default league for the app
FANTRAX_DEFAULT_LEAGUE_ID = "ifa1anexmdgtlk9s"

# All available league IDs
FANTRAX_LEAGUE_IDS = [
	"ifa1anexmdgtlk9s",
	"6zeydg0cm03y4myx",
	"8iozphpglhqp92bj",
	"sq5r3uxjl4nbtttj",
	"ajaqbnoekuklk4k8",
	"x6k2h5xim4rixppo"
]

# League names (corresponding to IDs above)
FANTRAX_LEAGUE_NAMES = [
	"Mr Squidward's Gay Layup Line",
	"Mr Squidwards 69",
	"Mister Squidward's N Word Pass",
	"Mister Squidward Travels",
	"Mister Squidward's Step Back 3",
	"Mr Squidward's 69 (2)"
]

# Create a mapping of ID to name
LEAGUE_ID_TO_NAME = dict(zip(FANTRAX_LEAGUE_IDS, FANTRAX_LEAGUE_NAMES))
LEAGUE_NAME_TO_ID = dict(zip(FANTRAX_LEAGUE_NAMES, FANTRAX_LEAGUE_IDS))
