"""Built-in category families. Importing this package registers every family (see `category.category_family`).

| family    | kind of work                    | relation to the rule | split |
|-----------|---------------------------------|----------------------|-------|
| transform | data transformation             | exemplified          | train |
| dedupe    | data quality / entity resolution| exemplified          | train |
| schedule  | scheduling                      | specified            | train |
| allocate  | resource allocation             | specified            | train |
| quote     | pricing under uncertain demand  | experiential         | train |
| retrieval | information retrieval           | exemplified          | val   |
| discover  | content discovery               | specified            | val   |
| netrel    | network reliability             | experiential         | test  |
| coord     | organisational coordination     | specified            | test  |
| triage    | classification / routing        | exemplified          | test  |

The three test families are held out entirely: no training or validation world contains a category of theirs.
"""
from . import allocate, coord, dedupe, discover, netrel, quote, retrieval, schedule, transform, triage  # noqa: F401
