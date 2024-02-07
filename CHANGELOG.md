# Changelog

## [v2.0.0](https://github.com/klarna-incubator/mleko/releases/tag/v2.0.0) (2024-02-07)

### ⛔️ BREAKING CHANGES

- **pipeline:** Refactor `PipelineStep` to use `TypedDict` for both inputs and outputs. ([`2eb623c`](https://github.com/klarna-incubator/mleko/commit/2eb623c593deab9fc87d3accdc7dcc8f5f600f97))

### 🐛 Bug Fixes

- **data cleaning:** Rename empty column name to `_empty` to prevent `vaex` crashes. ([`da72b75`](https://github.com/klarna-incubator/mleko/commit/da72b757f8395112bd0a17644af09547d5c72c0a))
- **data cleaning:** Cast boolean columns to `int8` during cleaning to reduce label encoding needs. ([`d94f7c9`](https://github.com/klarna-incubator/mleko/commit/d94f7c9f1c8f50a5b482668f85bb7382845092f0))
- Added reserved keyword column name replacement to prevent evaluation errors from `vaex`. ([`3969ffd`](https://github.com/klarna-incubator/mleko/commit/3969ffd4974854ee24d5dd1fce10f5169ff0e36e))

### 🛠️ Code Refactoring

- Improve error logging messages, and update codebase to new `black` format. ([`a29ad45`](https://github.com/klarna-incubator/mleko/commit/a29ad45156cad00e9cd01b09d677390187fc24d9))
- **cache:** Break out cache handler retrieval method. ([`aba9e41`](https://github.com/klarna-incubator/mleko/commit/aba9e4158bc76b9fa8b2444ae26fd6889ec832e3))

### 🤖 Continous Integration

- Remove TypeGuard and PyUpgrade from build and pre-commit. ([`d374406`](https://github.com/klarna-incubator/mleko/commit/d374406f4f595dd4b2fadd2d95bd7b9543f17d48))
- Add custom template for release notes to follow changelog structure. ([`30518c0`](https://github.com/klarna-incubator/mleko/commit/30518c066cbe1c8c8eabb3918531512dc4e37069))

## [v1.2.6](https://github.com/klarna-incubator/mleko/releases/tag/v1.2.6) (2024-01-25)

### 🐛 Bug Fixes

- Bump patch release. ([`ff5f94e`](https://github.com/klarna-incubator/mleko/commit/ff5f94e49df12db51cf18c22708a6dfeab49a942))

## [v1.2.5](https://github.com/klarna-incubator/mleko/releases/tag/v1.2.5) (2024-01-25)

### 🐛 Bug Fixes

- Fix `CHANGELOG.md` template location ([`141c9b7`](https://github.com/klarna-incubator/mleko/commit/141c9b7a9ea3e785308f6ed821ac799d60d163a1))

## [v1.2.4](https://github.com/klarna-incubator/mleko/releases/tag/v1.2.4) (2024-01-25)

### 🐛 Bug Fixes

- Trigger patch release. ([`7269dca`](https://github.com/klarna-incubator/mleko/commit/7269dcab2ce7e19fc2809a5485c1f1bdd33f2db2))

### 🏗️ Build

- **semantic versioning:** Update `CHANGELOG.md` template and semantic versioning logic. ([`1727e09`](https://github.com/klarna-incubator/mleko/commit/1727e097861c0f23520380a2b5d526d905861dee))

## [v1.2.3](https://github.com/klarna-incubator/mleko/releases/tag/v1.2.3) (2024-01-25)

### 🐛 Bug Fixes

- Remove coverage from workflow ([`09eb09d`](https://github.com/klarna-incubator/mleko/commit/09eb09da4fae09716c1fa37ebbc91dc532d7ed67))

## [v1.2.2](https://github.com/klarna-incubator/mleko/releases/tag/v1.2.2) (2024-01-25)

### 🐛 Bug Fixes

- Switch to trusted publishing ([`e84712d`](https://github.com/klarna-incubator/mleko/commit/e84712deb1775c4689a8d89b5bb54f0461494145))

## [v1.2.1](https://github.com/klarna-incubator/mleko/releases/tag/v1.2.1) (2024-01-25)

### 🐛 Bug Fixes

- Experiment with semantic versioning ([`0942196`](https://github.com/klarna-incubator/mleko/commit/0942196578df30f27c3d31435f5d2dc0b3c5775e))

### 🏗️ Build

- 🚧 Upgrade `python-gitlab` to `4.4.0` ([`15fff07`](https://github.com/klarna-incubator/mleko/commit/15fff07cae12e57a8bb7e30a2d4394ecee424e98))
- 🚧 Fix failing builds ([`79f7d95`](https://github.com/klarna-incubator/mleko/commit/79f7d959691924a1a8b2e68d462aa448969058da))

## [v1.2.0](https://github.com/klarna-incubator/mleko/releases/tag/v1.2.0) (2023-10-09)

### 🐛 Bug Fixes

- **logging:** 🐛 Fix LGBM logging routing to correct log level. ([`0e5fa77`](https://github.com/klarna-incubator/mleko/commit/0e5fa77ae14ccd27eb9b435ec8a73815fe764350))

### 🎨 Style

- remove unnecessary blank lines ([`a06edf2`](https://github.com/klarna-incubator/mleko/commit/a06edf23e865e6bfb95f9f3d13611a24de12b4ba))
- ✏️ Improve logging of `CSVToVaexConverter` and fix typo in `write_vaex_dataframe`. ([`197e56a`](https://github.com/klarna-incubator/mleko/commit/197e56ad4af59da2a08762c70b8e7ae5c5e350e2))

### 🏗️ Build

- 🔒️ Bump `gitpython` to resolve CVE-2023-41040 and CVE-2023-40590. ([`79627bd`](https://github.com/klarna-incubator/mleko/commit/79627bda5228d0470b2a3ca4613ae7636d79f4b6))

## [v1.1.0](https://github.com/klarna-incubator/mleko/releases/tag/v1.1.0) (2023-09-27)

### 🧪 Tests

- **tuning:** 🧪 Add test cases for `TuneStep`. ([`d811c7d`](https://github.com/klarna-incubator/mleko/commit/d811c7d4f60b4440b669cf723d3b85ee962efc97))

## [v1.0.0](https://github.com/klarna-incubator/mleko/releases/tag/v1.0.0) (2023-09-20)

### ⛔️ BREAKING CHANGES

- 📝 Improve `README.md` with more up to date information. ([`b388b59`](https://github.com/klarna-incubator/mleko/commit/b388b59a9c8da0df56da532a0cc17b347089a146))

## [v0.8.1](https://github.com/klarna-incubator/mleko/releases/tag/v0.8.1) (2023-09-07)

### 🐛 Bug Fixes

- **config:** 🐛 Fix readthedocs build to only generate html. ([`13fc207`](https://github.com/klarna-incubator/mleko/commit/13fc207adbba82fa9cc2d4dcad91038aecafa34f))

## [v0.8.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.8.0) (2023-09-06)

### 🐛 Bug Fixes

- **feature selection:** 🐛 Add `DataSchema` as partial return from all `fit` methods in feature selectors. ([`ebf2484`](https://github.com/klarna-incubator/mleko/commit/ebf2484fbaac7bbe067ab1491964bc50f4d7d711))

### 🛠️ Code Refactoring

- **cache:** 🚸 Replace `disable_cache` with a check if `cache_size=0` for `LRUCacheMixin`. ([`cfd7592`](https://github.com/klarna-incubator/mleko/commit/cfd759297d2d0e3b21f0f79827baab7c8b882784))

## [v0.7.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.7.0) (2023-07-11)

### 🐛 Bug Fixes

- **data cleaning:** 🐛 Switched to HDF5 as file format for faster I/O and better SageMaker support. ([`61f9e42`](https://github.com/klarna-incubator/mleko/commit/61f9e42ef215485b16d83be9726040b829d5a3e7))

## [v0.6.1](https://github.com/klarna-incubator/mleko/releases/tag/v0.6.1) (2023-06-30)

### 🐛 Bug Fixes

- **data cleaning:** 🐛 Fix date32/64[day] not converted to datetime. ([`98f4b26`](https://github.com/klarna-incubator/mleko/commit/98f4b26ef5eab26e47cdcc1fd0f31928e063d49a))
- **data source:** 🐛 Fix bug where S3 buckets with no manifest caused crash. ([`9078845`](https://github.com/klarna-incubator/mleko/commit/90788454dc8b2c1665f79746ae1392a8d4d067dd))

### 🏗️ Build

- **config:** 🔧 Switch mypy for pyright and update configuration. ([`5631aed`](https://github.com/klarna-incubator/mleko/commit/5631aedd059bdd6aed2c8e78d09629d8894e5cb8))

## [v0.6.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.6.0) (2023-06-26)

## [v0.5.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.5.0) (2023-06-17)

### 🛠️ Code Refactoring

- 💫 Switch to tqdm.auto to prevent breaking in Jupyter notebooks. ([`dc139cf`](https://github.com/klarna-incubator/mleko/commit/dc139cf3844d1a6beb7b5682b44275fd2f9ef2cf))

### 🧪 Tests

- ✅ Now _get_local_filenames returns a sorted list of filenames to ensure stability. ([`774e8eb`](https://github.com/klarna-incubator/mleko/commit/774e8eb2a38e5904a170473c56523926b3acffb4))

## [v0.4.2](https://github.com/klarna-incubator/mleko/releases/tag/v0.4.2) (2023-06-11)

### 🛠️ Code Refactoring

- ➖ Remove pandas dependency. ([`40e264c`](https://github.com/klarna-incubator/mleko/commit/40e264c0604793e860887bbbd03b5c374fa55162))

### 🤖 Continous Integration

- **semantic versioning:** 👷 Add more sections to changelog based on conventional commit categories. ([`e5b1594`](https://github.com/klarna-incubator/mleko/commit/e5b15944ae98793273ed407b13867abf68783f36))

## [v0.4.1](https://github.com/klarna-incubator/mleko/releases/tag/v0.4.1) (2023-06-04)

### 🐛 Bug Fixes

- **feature selection:** 🐛 Fix `FeatureSelector` cache to use tuple in… (#60) ([`758cf5e`](https://github.com/klarna-incubator/mleko/commit/758cf5ee770567e89dfac6dc06a8539318492feb))
- **feature selection:** 🐛 Fix `FeatureSelector` cache to use tuple instead of frozenset to have stable fingerprint. ([`cd82417`](https://github.com/klarna-incubator/mleko/commit/cd824177e252ae00af4d2ed8cc50b607c4a4928f))

## [v0.4.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.4.0) (2023-06-03)

### 🐛 Bug Fixes

- 🐛 Fix typeguard breaking changes causing build to fail. ([`66c6a8e`](https://github.com/klarna-incubator/mleko/commit/66c6a8e24e04815f35dc18de5b3ccd20589fe79d))

### 🛠️ Code Refactoring

- 🔥 Unify dataset subpackage naming to verbs and modules to nouns. ([`3ffb909`](https://github.com/klarna-incubator/mleko/commit/3ffb909806ceae8162ac31719bb228e9238a334d))
- 🔥 Rename subpackages in dataset to singular variant. ([`51a8297`](https://github.com/klarna-incubator/mleko/commit/51a829707d41dcb66f7fba3d5c8b34bad3618225))
- 🔥 Refactor entire project to improve maintainability. ([`dd1d22c`](https://github.com/klarna-incubator/mleko/commit/dd1d22c52b678d56cb4750a96997c897d1a7f35c))

## [v0.3.1](https://github.com/klarna-incubator/mleko/releases/tag/v0.3.1) (2023-05-21)

### 🐛 Bug Fixes

- :bug: Added notes to pipeline step docstrings. ([`d94f899`](https://github.com/klarna-incubator/mleko/commit/d94f89921befafe6a81540c996626e7a849fd260))

### 🛠️ Code Refactoring

- **data source:** :bug: Added note to the KaggleDataSource init docstring. ([`d5f12d3`](https://github.com/klarna-incubator/mleko/commit/d5f12d32eb1492bc3c2566ff16e6fd370301754b))

### 🤖 Continous Integration

- :rocket: Removed semantic PR workflow and updated test workflow to not run on release commits. ([`8138745`](https://github.com/klarna-incubator/mleko/commit/8138745913457e4e69858c180b2ebacac9f40861))

## [v0.3.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.3.0) (2023-05-21)

### 🐛 Bug Fixes

- **data splitting:** :bug: Added notes and examples to splitters docstrings. ([`d162c86`](https://github.com/klarna-incubator/mleko/commit/d162c8611c0284d0a83e4b5ad9664f964351d194))
- **pipeline:** :bug: Updated some docstrings. ([`56b36fd`](https://github.com/klarna-incubator/mleko/commit/56b36fd70030b2d39351b74b3c0e4dce5bd9a06c))

### 🤖 Continous Integration

- :rocket: Updated release to only trigger if the commit message does not contain chore(release). ([`c9f3f3f`](https://github.com/klarna-incubator/mleko/commit/c9f3f3f4a666bcbac44bf86478487057e95609e0))

## [v0.2.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.2.0) (2023-05-21)

## [v0.1.3](https://github.com/klarna-incubator/mleko/releases/tag/v0.1.3) (2023-05-13)

### 🐛 Bug Fixes

- **cache:** :bug: Cache modules exposed in subpackage __init__. ([`fd65e9d`](https://github.com/klarna-incubator/mleko/commit/fd65e9df668a0003543a1fa2e3260f723a87285f))

## [v0.1.2](https://github.com/klarna-incubator/mleko/releases/tag/v0.1.2) (2023-05-13)

### 🐛 Bug Fixes

- **cache:** :bug: Fixed LRUCacheMixin eviction test case. ([`ce5bfc1`](https://github.com/klarna-incubator/mleko/commit/ce5bfc17aabaf858778eb42ef40c9f38a9e2ae97))
- :bug: Temporarely disabled failing tests for cache. ([`9c17960`](https://github.com/klarna-incubator/mleko/commit/9c17960185606b6e9d150154bc4b6aac6592a7cc))

## [v0.1.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.1.0) (2023-05-12)

### 🐛 Bug Fixes

- **cache:** :bug: Fixed test by not testing it... ([`e3a0ce9`](https://github.com/klarna-incubator/mleko/commit/e3a0ce91da26bf424b199105be164c33b3a6b354))
- **cache:** :bug: Try logging using assert to fix GH issue ([`5e247ec`](https://github.com/klarna-incubator/mleko/commit/5e247ec2de355148b1710a1691404f8cffc4bd31))
- **cache:** :bug: Attempting to fix test case failing in GH actions. ([`4892591`](https://github.com/klarna-incubator/mleko/commit/4892591e9ab6244ec6494d9fc9532eeeb8f12c68))
- **cache:** :bug: LRUCacheMixin now relies on file modification time instead of access time due to system limitations. ([`127d657`](https://github.com/klarna-incubator/mleko/commit/127d6578715f5e852e3638d8a037d072991034d7))
- :bug: Fixed docstrings for private methods in KaggleDataSource and removed xdoctest from build steps ([`bb55cf5`](https://github.com/klarna-incubator/mleko/commit/bb55cf55c149a8fdec3458799e987969174b5207))
