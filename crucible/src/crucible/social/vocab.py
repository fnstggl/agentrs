"""Topic vocabularies for the text that population creators write (workstream C2).

Population creators write real text artifacts (posts and articles) from a mixture of the topics below plus
common function words. Nothing downstream reads a topic label: participants respond to the encoded text of the
artifact (`crucible.social.textrep`), so two posts about the same subject are similar only because they share words.
The word lists are ordinary English vocabulary chosen so that topics overlap a little (for example `match` in sport
and dating, `market` in finance and food), as real subjects do.
"""
from __future__ import annotations

TOPICS: dict[str, tuple[str, ...]] = {
    "cooking": ("recipe", "garlic", "oven", "simmer", "pasta", "sauce", "bake", "flour", "butter", "onion", "roast",
                "spice", "kitchen", "dinner", "soup", "knife", "pan", "salt", "pepper", "dough", "herbs", "grill",
                "taste", "market"),
    "football": ("goal", "striker", "league", "match", "keeper", "penalty", "coach", "season", "transfer", "derby",
                 "defender", "midfield", "referee", "stadium", "fans", "kickoff", "tactics", "winger", "cup", "pitch",
                 "score", "foul", "offside", "captain"),
    "programming": ("python", "compiler", "function", "bug", "refactor", "library", "commit", "branch", "deploy",
                    "server", "database", "query", "thread", "memory", "pointer", "rust", "kernel", "test", "api",
                    "framework", "debug", "release", "package", "code"),
    "music": ("guitar", "album", "chord", "melody", "drums", "concert", "lyrics", "rhythm", "band", "vinyl", "studio",
              "piano", "bass", "tour", "singer", "jazz", "remix", "tempo", "festival", "song", "harmony", "beat",
              "playlist", "stage"),
    "finance": ("stocks", "bond", "inflation", "dividend", "portfolio", "market", "interest", "savings", "budget",
                "mortgage", "index", "fund", "rates", "credit", "tax", "earnings", "yield", "debt", "pension",
                "invest", "broker", "risk", "loan", "cash"),
    "gardening": ("seeds", "compost", "tomato", "soil", "prune", "roses", "greenhouse", "watering", "mulch", "bulbs",
                  "harvest", "weeds", "orchard", "spade", "seedling", "bloom", "hedge", "lawn", "herbs", "fertilizer",
                  "shade", "planting", "vegetable", "garden"),
    "fitness": ("workout", "squat", "protein", "running", "marathon", "cardio", "stretch", "muscle", "gym", "sprint",
                "yoga", "deadlift", "recovery", "endurance", "calories", "plank", "training", "pace", "weights",
                "mobility", "coach", "sweat", "reps", "core"),
    "travel": ("flight", "passport", "hostel", "beach", "itinerary", "luggage", "train", "island", "museum", "visa",
               "backpack", "hotel", "mountains", "village", "ferry", "airport", "route", "map", "journey", "temple",
               "coast", "border", "hike", "tour"),
    "gaming": ("console", "quest", "level", "boss", "multiplayer", "controller", "speedrun", "loot", "pixel",
               "strategy", "esports", "dungeon", "patch", "server", "character", "arcade", "raid", "puzzle", "score",
               "stream", "mod", "co-op", "release", "save"),
    "science": ("quantum", "experiment", "galaxy", "molecule", "gene", "physics", "telescope", "cell", "evolution",
                "particle", "climate", "fossil", "laboratory", "hypothesis", "orbit", "protein", "energy", "atom",
                "theory", "data", "neuron", "species", "planet", "research"),
    "parenting": ("toddler", "bedtime", "school", "nappy", "teenager", "homework", "sibling", "playground", "baby",
                  "tantrum", "lunchbox", "routine", "nursery", "family", "screen", "reading", "kindness", "chores",
                  "allowance", "holiday", "sleep", "milestone", "teacher", "patience"),
    "fashion": ("dress", "denim", "sneakers", "fabric", "runway", "jacket", "style", "vintage", "tailor", "collection",
                "silk", "boots", "outfit", "trend", "wardrobe", "designer", "colour", "knit", "accessories", "season",
                "thrift", "shirt", "fit", "pattern"),
    "cars": ("engine", "torque", "electric", "battery", "gearbox", "tyres", "horsepower", "sedan", "charging",
             "brakes", "diesel", "hybrid", "racing", "mileage", "garage", "suspension", "dashboard", "turbo", "road",
             "license", "range", "motor", "service", "drive"),
    "pets": ("puppy", "kitten", "vet", "leash", "litter", "breed", "adoption", "treats", "grooming", "aquarium",
             "parrot", "walkies", "collar", "rescue", "hamster", "training", "bark", "purr", "kennel", "fur", "toy",
             "vaccine", "food", "paws"),
    "history": ("empire", "medieval", "revolution", "archive", "dynasty", "battle", "castle", "treaty", "pharaoh",
                "colony", "monarch", "manuscript", "ruins", "century", "republic", "trade", "war", "museum", "legend",
                "kingdom", "artifact", "chronicle", "siege", "map"),
    "film": ("director", "trailer", "cinema", "actor", "screenplay", "sequel", "premiere", "camera", "scene", "drama",
             "comedy", "horror", "award", "editing", "soundtrack", "cast", "plot", "studio", "festival", "review",
             "animation", "frame", "documentary", "series"),
    "crypto": ("bitcoin", "wallet", "blockchain", "token", "exchange", "mining", "ledger", "defi", "staking", "coin",
               "volatility", "nft", "altcoin", "hash", "protocol", "smart", "contract", "airdrop", "chain", "market",
               "risk", "keys", "node", "fees"),
    "diy": ("drill", "plank", "shelf", "paint", "plaster", "screws", "workbench", "tiles", "sanding", "glue",
            "cabinet", "hammer", "renovation", "insulation", "wiring", "plumbing", "saw", "varnish", "measure",
            "repair", "deck", "fence", "toolbox", "project"),
    "books": ("novel", "author", "chapter", "library", "poetry", "fiction", "memoir", "paperback", "series",
              "publisher", "plot", "character", "reading", "review", "classic", "thriller", "fantasy", "bookshop",
              "translation", "essay", "prize", "story", "audiobook", "shelf"),
    "health": ("sleep", "nutrition", "doctor", "vitamin", "stress", "therapy", "immune", "clinic", "symptoms",
               "hydration", "diet", "anxiety", "posture", "screening", "habit", "wellbeing", "heart", "breathing",
               "allergy", "medicine", "rest", "checkup", "balance", "energy"),
    "art": ("painting", "canvas", "sketch", "gallery", "sculpture", "watercolor", "portrait", "ink", "studio",
            "brush", "exhibition", "mural", "pottery", "charcoal", "abstract", "colour", "frame", "print", "artist",
            "drawing", "clay", "texture", "collage", "light"),
    "politics": ("election", "parliament", "policy", "vote", "minister", "campaign", "debate", "budget", "reform",
                 "council", "law", "party", "senate", "protest", "tax", "speech", "coalition", "ballot", "rights",
                 "summit", "treaty", "poll", "mayor", "court"),
    "space": ("rocket", "launch", "astronaut", "orbit", "mars", "moon", "satellite", "nasa", "spacecraft", "comet",
              "nebula", "lander", "station", "gravity", "telescope", "asteroid", "probe", "mission", "booster",
              "capsule", "solar", "eclipse", "planet", "stars"),
    "startups": ("founder", "funding", "pitch", "investor", "product", "launch", "customers", "growth", "revenue",
                 "equity", "seed", "startup", "pivot", "market", "hiring", "churn", "pricing", "users", "subscription",
                 "runway", "valuation", "team", "metrics", "build"),
}

TOPIC_NAMES: tuple[str, ...] = tuple(TOPICS)

FUNCTION_WORDS: tuple[str, ...] = (
    "the", "a", "and", "of", "to", "in", "is", "it", "this", "that", "with", "for", "on", "my", "your", "we", "you",
    "i", "was", "are", "be", "have", "so", "but", "about", "just", "really", "today", "here", "new", "how", "why",
    "what", "best", "first", "week", "time", "make", "think", "try", "love", "good", "great", "people", "some",
)

# words that carry no subject: excluded from the content representation and from search terms
STOPWORDS = frozenset(FUNCTION_WORDS)
