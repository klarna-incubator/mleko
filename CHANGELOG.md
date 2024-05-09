# Changelog

## [v4.0.0](https://github.com/klarna-incubator/mleko/releases/tag/v4.0.0) (2024-05-09)

### ⛔️ BREAKING CHANGES

- **exporter:** Add `S3Exporter` that implements cached S3 exporting of files from the local disk. ([`d17b2d2`](https://github.com/klarna-incubator/mleko/commit/d17b2d20e93d0542f25e2142e4dc61c104f7e0b8))
- **exporter:** Add `BaseExporter` and `LocalExporter` implementations that support exporting data to disk, along with corresponding `Pipeline` steps. ([`6ce13cf`](https://github.com/klarna-incubator/mleko/commit/6ce13cf5bff65c2da429f1db80fc9612f1f9c223))

### ✨ Features

- **exporter:** Add `LocalManifest` support for `LocalExporter` which simplifies caching logic and enables S3 manifest translations. ([`2199ff0`](https://github.com/klarna-incubator/mleko/commit/2199ff0a5d2bf347ef25898c28f94c59ec4b210e))
- **exporter:** Add support for multiple data export using `LocalExporter`. ([`ff988b6`](https://github.com/klarna-incubator/mleko/commit/ff988b683ca59782212d34f915e9e55587f6a9e8))
- **data source:** Add support for reading manifest files from S3 buckets in `S3Ingester`. ([`9c68a9b`](https://github.com/klarna-incubator/mleko/commit/9c68a9b107a0436cbdace8cdbd41f839783f9369))
- **pipeline:** Add `disable_cache` parameter to `Pipeline` execution. ([`da1e31a`](https://github.com/klarna-incubator/mleko/commit/da1e31a90a2e1de58f15dd0df91309bc81f10427))

### 🐛 Bug Fixes

- **data cleaning:** Fix newline characters breaking CSV reading using Arrow. ([`3a7e594`](https://github.com/klarna-incubator/mleko/commit/3a7e59426ff8575f4ece679fd1468edec3173cfe))
- **tuning:** Delete logging of storage URI to minimize risk of accidentally logging credentials. ([`054692d`](https://github.com/klarna-incubator/mleko/commit/054692d5508018fd77ea82c05fe27a2dd2f229b3))

### 🛠️ Code Refactoring

- **data source:** Extract shared S3 logic to `utils` which can be then used by `S3Exporter`. ([`97a7974`](https://github.com/klarna-incubator/mleko/commit/97a79748996aede230507fa88d09839d00d2029e))

## [v3.2.0](https://github.com/klarna-incubator/mleko/releases/tag/v3.2.0) (2024-04-18)

### ✨ Features

- **tuning:** Add support for `RDSStorage` using the `OptunaTuner` ([`cc06ddd`](https://github.com/klarna-incubator/mleko/commit/cc06dddd3e9c6e0ac8968319ef64881039a1fec4))

### 🐛 Bug Fixes

- **data source:** Fix bug where `dataset_id` consisting of path components would break local metadata file creation ([`17c4866`](https://github.com/klarna-incubator/mleko/commit/17c48669f22ff449fad28528340ad8f5221361bb))
- **model:** Add `verbosity` parameter to `BaseModel` to set log level in the base class. ([`0a3828f`](https://github.com/klarna-incubator/mleko/commit/0a3828f246c82ba1f8257da53de0743d9b92c3fa))

## [v3.1.0](https://github.com/klarna-incubator/mleko/releases/tag/v3.1.0) (2024-04-12)

### ✨ Features

- **model:** Add optional memoization to datasets during model training. (#209) ([`2ca4465`](https://github.com/klarna-incubator/mleko/commit/2ca44651b793ca0b2c9a38bf8ac99934da07760a))
- **model:** Add optional memoization to datasets during model training. ([`6a955dc`](https://github.com/klarna-incubator/mleko/commit/6a955dca44bc1df139f5dde2e34bb97d12ae8de8))

## [v3.0.0](https://github.com/klarna-incubator/mleko/releases/tag/v3.0.0) (2024-04-05)

### ⛔️ BREAKING CHANGES

- **model:** Update `LGBMModel` to use dependency injection, now expects a `lightgbm.LGBMModel` as argument. ([`7250f34`](https://github.com/klarna-incubator/mleko/commit/7250f344304eadc013ed54b40c7a1e017133833c))

### 🐛 Bug Fixes

- Switch `vaex` file format to `Arrow` instead of `HDF5` for better type support. ([`ac8e500`](https://github.com/klarna-incubator/mleko/commit/ac8e5008617ecc2185f9cb6927b578d24807265b))
- **data cleaning:** Fix bug where boolean columns are stored as numerical in the data schema due to `int8` conversion. ([`da358d8`](https://github.com/klarna-incubator/mleko/commit/da358d850313fba116cb69fe554f727de0a19380))

## [v2.2.0](https://github.com/klarna-incubator/mleko/releases/tag/v2.2.0) (2024-03-22)

### ✨ Features

- **filter:** Add `ImblearnResamplingFilter` which is a wrapper for `imblearn` over- and under-samplers. ([`77a3d7d`](https://github.com/klarna-incubator/mleko/commit/77a3d7d1a86c37ff91725afb12e6d2b1a6f1e649))
- **filter:** Add `ExpressionFilter` and base class for simple DataFrame filtering using `vaex` expressions. ([`dc679ff`](https://github.com/klarna-incubator/mleko/commit/dc679ffc92d4ea2f4e064f51076b3edea61fd838))
- **cache:** Add `disable_cache` argument to all cached functions to completely bypass all caching functionality. ([`fbdfc5d`](https://github.com/klarna-incubator/mleko/commit/fbdfc5dd94e5aaf440268efecf3ea1c4c6762d86))

### 📝 Documentation

- Update `CHANGELOG.md` format to include missing categories. ([`d97b32c`](https://github.com/klarna-incubator/mleko/commit/d97b32c9e825f59518b214d8b8338a470fcee7d4))

## [v2.1.0](https://github.com/klarna-incubator/mleko/releases/tag/v2.1.0) (2024-02-24)

### ✨ Features

- Update Titanic dataset to `mleko` 2.0 API. ([`62bf991`](https://github.com/klarna-incubator/mleko/commit/62bf991fa1324fefe5d83fc0ea1c36f62f2cdbb3))
- **tuning:** Add `optuna-dashboard` support to `OptunaTuner` including automatically generated experiment notes. ([`29d81c2`](https://github.com/klarna-incubator/mleko/commit/29d81c288b8dffcc9374bfa813e987bda80228df))
- **transformer:** Improve flexibility of `LabelEncoderTransformer` by adding optional null encoding and manual dictionary mapping. ([`f7b30a9`](https://github.com/klarna-incubator/mleko/commit/f7b30a9a17388b27163fd9addae378935974ddab))
- Set `cache_directory` as optional argument, with custom default locations. ([`08e8777`](https://github.com/klarna-incubator/mleko/commit/08e8777aac8fca0429e0cdc887426fbc25907bc2))

### 🐛 Bug Fixes

- **data cleaning:** Fix `meta_columns` not being forcefully cast to correct data type in `CSVToVaexConverter`. ([`b42b9ed`](https://github.com/klarna-incubator/mleko/commit/b42b9ed7a8592a4c3038bb3dc5b7dd512e4be2e6))

### 📝 Documentation

- Update year in Copyright in README.md (#192) ([`eeb56e1`](https://github.com/klarna-incubator/mleko/commit/eeb56e1e04f34aff44527da463717f8cd41a2302))

### 🧪 Tests

- Fix test cases generating cache directory outside temporary directory. ([`ba57fbf`](https://github.com/klarna-incubator/mleko/commit/ba57fbfc838fd213446524a652cf235e002dde25))

## [v2.0.0](https://github.com/klarna-incubator/mleko/releases/tag/v2.0.0) (2024-02-07)

### ⛔️ BREAKING CHANGES

- **pipeline:** Refactor `PipelineStep` to use `TypedDict` for both inputs and outputs. ([`2eb623c`](https://github.com/klarna-incubator/mleko/commit/2eb623c593deab9fc87d3accdc7dcc8f5f600f97))

### ✨ Features

- **model:** Refactor validation_dataframe parameter in BaseModel and LGBMModel to be optional. ([`d18ed29`](https://github.com/klarna-incubator/mleko/commit/d18ed291926e1969ba8d4163964f443956626ae2))
- **cache:** Add cache support for `None` returns on fields using cache handlers not equipped to process None. ([`a489996`](https://github.com/klarna-incubator/mleko/commit/a489996258884d1a584fc33388617ae758c7c009))
- **model:** Add support for custom evaluation function in LGBMModel. ([`4e70a55`](https://github.com/klarna-incubator/mleko/commit/4e70a55304fd6a49109add2e6d987bb77f3a9eaf))

### 🐛 Bug Fixes

- **data cleaning:** Rename empty column name to `_empty` to prevent `vaex` crashes. ([`da72b75`](https://github.com/klarna-incubator/mleko/commit/da72b757f8395112bd0a17644af09547d5c72c0a))
- **data cleaning:** Cast boolean columns to `int8` during cleaning to reduce label encoding needs. ([`d94f7c9`](https://github.com/klarna-incubator/mleko/commit/d94f7c9f1c8f50a5b482668f85bb7382845092f0))
- Added reserved keyword column name replacement to prevent evaluation errors from `vaex`. ([`3969ffd`](https://github.com/klarna-incubator/mleko/commit/3969ffd4974854ee24d5dd1fce10f5169ff0e36e))

### 🛠️ Code Refactoring

- Improve error logging messages, and update codebase to new `black` format. ([`a29ad45`](https://github.com/klarna-incubator/mleko/commit/a29ad45156cad00e9cd01b09d677390187fc24d9))
- **cache:** Break out cache handler retrieval method. ([`aba9e41`](https://github.com/klarna-incubator/mleko/commit/aba9e4158bc76b9fa8b2444ae26fd6889ec832e3))

### 📝 Documentation

- Refactor mleko package documentation to format bullet list correctly. ([`76ee895`](https://github.com/klarna-incubator/mleko/commit/76ee895fe284aff3409501a8628af61f18392364))

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

### ✨ Features

- **data source:** ✨ Add support for pattern matching in `*Ingester` and add `LocalManifest` to index fetched files. ([`75974a4`](https://github.com/klarna-incubator/mleko/commit/75974a40aa1af5e21a4dedc7e0d05be2153ec7aa))

### 🐛 Bug Fixes

- **logging:** 🐛 Fix LGBM logging routing to correct log level. ([`0e5fa77`](https://github.com/klarna-incubator/mleko/commit/0e5fa77ae14ccd27eb9b435ec8a73815fe764350))

### 🎨 Style

- remove unnecessary blank lines ([`a06edf2`](https://github.com/klarna-incubator/mleko/commit/a06edf23e865e6bfb95f9f3d13611a24de12b4ba))
- ✏️ Improve logging of `CSVToVaexConverter` and fix typo in `write_vaex_dataframe`. ([`197e56a`](https://github.com/klarna-incubator/mleko/commit/197e56ad4af59da2a08762c70b8e7ae5c5e350e2))

### 🏗️ Build

- 🔒️ Bump `gitpython` to resolve CVE-2023-41040 and CVE-2023-40590. ([`79627bd`](https://github.com/klarna-incubator/mleko/commit/79627bda5228d0470b2a3ca4613ae7636d79f4b6))

## [v1.1.0](https://github.com/klarna-incubator/mleko/releases/tag/v1.1.0) (2023-09-27)

### ✨ Features

- **tuning:** ✨ Add hyperparameter tuning functionality, initially including `OptunaTuner`. ([`be38c07`](https://github.com/klarna-incubator/mleko/commit/be38c075e269ed2e42560dfa26ac71ab08448444))

### 🧪 Tests

- **tuning:** 🧪 Add test cases for `TuneStep`. ([`d811c7d`](https://github.com/klarna-incubator/mleko/commit/d811c7d4f60b4440b669cf723d3b85ee962efc97))

## [v1.0.0](https://github.com/klarna-incubator/mleko/releases/tag/v1.0.0) (2023-09-20)

### ⛔️ BREAKING CHANGES

- 📝 Improve `README.md` with more up to date information. ([`b388b59`](https://github.com/klarna-incubator/mleko/commit/b388b59a9c8da0df56da532a0cc17b347089a146))

### ✨ Features

- **transformer:** ✨ Add `DataSchema` API to transformers `fit`, `transform` and `fit_transform`. ([`e053c85`](https://github.com/klarna-incubator/mleko/commit/e053c854238614ef23c6e8ff7eedc004cbeedf71))

### 📝 Documentation

- 📝 Add example notebook for `Titanic` dataset. ([`e651af9`](https://github.com/klarna-incubator/mleko/commit/e651af99d96f2d012faaf6ab5c73cc681f7a5b5b))

## [v0.8.1](https://github.com/klarna-incubator/mleko/releases/tag/v0.8.1) (2023-09-07)

### 🐛 Bug Fixes

- **config:** 🐛 Fix readthedocs build to only generate html. ([`13fc207`](https://github.com/klarna-incubator/mleko/commit/13fc207adbba82fa9cc2d4dcad91038aecafa34f))

## [v0.8.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.8.0) (2023-09-06)

### ✨ Features

- **model:** ✨ Add `LGBMModel` along with base class which can be extended for all types of future models. ([`b47a241`](https://github.com/klarna-incubator/mleko/commit/b47a2412e8925226d5612e4b57c521f562c51636))
- ✨ Add` DataSchema` which tracks dataset features throughout the pipeline and methods. ([`e03bd2c`](https://github.com/klarna-incubator/mleko/commit/e03bd2cbc9e75347eeea37f5f818d05a2548c3de))
- **feature selection:** ✨ Update `BaseFeatureSelector` and children to use the `fit`, `transform` and `fit_transform` pattern. ([`62e4dd1`](https://github.com/klarna-incubator/mleko/commit/62e4dd13a37a852f239c490d2693b20665095cf6))
- **transformer:** ✨ Add `fit`, `transform` and `fit_transform` to all `Transformers`, along with API and caching simplificatons. ([`5cc4ebc`](https://github.com/klarna-incubator/mleko/commit/5cc4ebc9da3b3f3aa5e8788d1237ac2c79ee4693))
- **cache:** ✨ Add `CacheHandler` which allows customization of read/write functions for each cached return value individually. ([`609e084`](https://github.com/klarna-incubator/mleko/commit/609e084e7c354bed5a298b3a3a1c3b033d3db71d))

### 🐛 Bug Fixes

- **feature selection:** 🐛 Add `DataSchema` as partial return from all `fit` methods in feature selectors. ([`ebf2484`](https://github.com/klarna-incubator/mleko/commit/ebf2484fbaac7bbe067ab1491964bc50f4d7d711))

### 🛠️ Code Refactoring

- **cache:** 🚸 Replace `disable_cache` with a check if `cache_size=0` for `LRUCacheMixin`. ([`cfd7592`](https://github.com/klarna-incubator/mleko/commit/cfd759297d2d0e3b21f0f79827baab7c8b882784))

## [v0.7.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.7.0) (2023-07-11)

### ✨ Features

- ✨ Add fit transform support to all `FeatureSelector` along with refactoring the `LRUCacheMixin`. ([`3df0601`](https://github.com/klarna-incubator/mleko/commit/3df06011085a1230c00250260374e5fd5d325fad))
- ✨ Add support for separate fitting and transforming inside the pipeline. ([`bb9b7a4`](https://github.com/klarna-incubator/mleko/commit/bb9b7a4ea9920d5588972a29e390bac4017b45af))

### 🐛 Bug Fixes

- **data cleaning:** 🐛 Switched to HDF5 as file format for faster I/O and better SageMaker support. ([`61f9e42`](https://github.com/klarna-incubator/mleko/commit/61f9e42ef215485b16d83be9726040b829d5a3e7))

## [v0.6.1](https://github.com/klarna-incubator/mleko/releases/tag/v0.6.1) (2023-06-30)

### 🐛 Bug Fixes

- **data cleaning:** 🐛 Fix date32/64[day] not converted to datetime. ([`98f4b26`](https://github.com/klarna-incubator/mleko/commit/98f4b26ef5eab26e47cdcc1fd0f31928e063d49a))
- **data source:** 🐛 Fix bug where S3 buckets with no manifest caused crash. ([`9078845`](https://github.com/klarna-incubator/mleko/commit/90788454dc8b2c1665f79746ae1392a8d4d067dd))

### 🏗️ Build

- **config:** 🔧 Switch mypy for pyright and update configuration. ([`5631aed`](https://github.com/klarna-incubator/mleko/commit/5631aedd059bdd6aed2c8e78d09629d8894e5cb8))

## [v0.6.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.6.0) (2023-06-26)

### ✨ Features

- **cache:** ✨ Add cache_group that can segment an instance cache into different isolated parts. (#66) ([`5fa8c9c`](https://github.com/klarna-incubator/mleko/commit/5fa8c9c39b96402daab25f12dabdcf95f425e066))
- **cache:** ✨ Add cache_group that can segment an instance cache into different isolated parts. ([`b5c3de5`](https://github.com/klarna-incubator/mleko/commit/b5c3de5f1397bfb9422590ba037f431c2eaad9ac))

## [v0.5.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.5.0) (2023-06-17)

### ✨ Features

- **transformer:** ✨ Add MinMaxScalerTransformer for normalizing numerical features. ([`9b26c00`](https://github.com/klarna-incubator/mleko/commit/9b26c00ecefa12929a2316309ba5b3e55022b5a5))
- **transformer:** ✨ Add MaxAbsScalerTransformer that scales numerical features. ([`1fd2a93`](https://github.com/klarna-incubator/mleko/commit/1fd2a9369bfb74eba3d0015024857070f31bbe34))
- **transformer:** ✨ Add CompositeTransformer for chaining together multiple transformers sequentially. ([`006d741`](https://github.com/klarna-incubator/mleko/commit/006d74157167e4ddb2c575ca897662de8a1c0d4d))
- **transformer:** ✨ Add LabelEncoderTransformer for ordinal encoding. ([`41a4c45`](https://github.com/klarna-incubator/mleko/commit/41a4c45dcb7a232f2a8f3af3a39e9bcf4bcb3929))
- **transformer:** ✨ Add FrequencyEncoderTransformer along with support for pipeline. ([`465e6db`](https://github.com/klarna-incubator/mleko/commit/465e6db3b2830e3cc3f996af9eacb36b8ccf8468))

### 🛠️ Code Refactoring

- 💫 Switch to tqdm.auto to prevent breaking in Jupyter notebooks. ([`dc139cf`](https://github.com/klarna-incubator/mleko/commit/dc139cf3844d1a6beb7b5682b44275fd2f9ef2cf))

### 🧪 Tests

- ✅ Now _get_local_filenames returns a sorted list of filenames to ensure stability. ([`774e8eb`](https://github.com/klarna-incubator/mleko/commit/774e8eb2a38e5904a170473c56523926b3acffb4))

## [v0.4.2](https://github.com/klarna-incubator/mleko/releases/tag/v0.4.2) (2023-06-11)

### 🚀 Performance improvements

- ⚡️ Optimize VarianceFeatureSelector when threshold is 0. ([`906dde3`](https://github.com/klarna-incubator/mleko/commit/906dde391d83bc3b07ac5a56bb690dba294e757f))

### 🛠️ Code Refactoring

- ➖ Remove pandas dependency. ([`40e264c`](https://github.com/klarna-incubator/mleko/commit/40e264c0604793e860887bbbd03b5c374fa55162))

### 🤖 Continous Integration

- **semantic versioning:** 👷 Add more sections to changelog based on conventional commit categories. ([`e5b1594`](https://github.com/klarna-incubator/mleko/commit/e5b15944ae98793273ed407b13867abf68783f36))

## [v0.4.1](https://github.com/klarna-incubator/mleko/releases/tag/v0.4.1) (2023-06-04)

### 🐛 Bug Fixes

- **feature selection:** 🐛 Fix `FeatureSelector` cache to use tuple in… (#60) ([`758cf5e`](https://github.com/klarna-incubator/mleko/commit/758cf5ee770567e89dfac6dc06a8539318492feb))
- **feature selection:** 🐛 Fix `FeatureSelector` cache to use tuple instead of frozenset to have stable fingerprint. ([`cd82417`](https://github.com/klarna-incubator/mleko/commit/cd824177e252ae00af4d2ed8cc50b607c4a4928f))

## [v0.4.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.4.0) (2023-06-03)

### ✨ Features

- **feature selection:** ✨ Add  that filters out invariant features. ([`798c261`](https://github.com/klarna-incubator/mleko/commit/798c26103b41fd32f71d16b7b8d40f647f2c849b))
- **feature selection:** ✨ Add `PearsonCorrelationFeatureSelector` which drops highly correlated features. ([`66e5cd2`](https://github.com/klarna-incubator/mleko/commit/66e5cd28a4621172ed1c3d27cfc430b89d7da321))
- **feature selection:** ✨ Add `CompositeFeatureSelector`, for chaining multiple feature selection steps on the same DataFrame. ([`3d75079`](https://github.com/klarna-incubator/mleko/commit/3d75079d1bc9ab1102a9edb6dbb545b03f43b4dd))
- **feature selection:** ✨ Add standard deviation feature selector. ([`c56177b`](https://github.com/klarna-incubator/mleko/commit/c56177bb7e25cbf763418707d09976290f659088))
- **feature selection:** ✨ Add missing rate feature selector. ([`d5ba8b5`](https://github.com/klarna-incubator/mleko/commit/d5ba8b57ae4102b2c5c424556d86640f2934dc46))

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

### ✨ Features

- new notes (#54) ([`21239f7`](https://github.com/klarna-incubator/mleko/commit/21239f7f7f26cb8fe7b419e81a3e7fe9dd3736fd))

### 🐛 Bug Fixes

- **data splitting:** :bug: Added notes and examples to splitters docstrings. ([`d162c86`](https://github.com/klarna-incubator/mleko/commit/d162c8611c0284d0a83e4b5ad9664f964351d194))
- **pipeline:** :bug: Updated some docstrings. ([`56b36fd`](https://github.com/klarna-incubator/mleko/commit/56b36fd70030b2d39351b74b3c0e4dce5bd9a06c))

### 🤖 Continous Integration

- :rocket: Updated release to only trigger if the commit message does not contain chore(release). ([`c9f3f3f`](https://github.com/klarna-incubator/mleko/commit/c9f3f3f4a666bcbac44bf86478487057e95609e0))

## [v0.2.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.2.0) (2023-05-21)

### ✨ Features

- add data splitting step (#53) ([`a668b1a`](https://github.com/klarna-incubator/mleko/commit/a668b1a0cd61b6fa1970f401f8209accd40e083f))

### 📝 Documentation

- Removed duplicate row. ([`5d77131`](https://github.com/klarna-incubator/mleko/commit/5d77131341fb9d241fb179115f1dbd51c9962059))
- Adding pre-commit check for conventional commits. ([`dd2076e`](https://github.com/klarna-incubator/mleko/commit/dd2076e38711368a94ba7b454ce830caf23ebf1d))

## [v0.1.3](https://github.com/klarna-incubator/mleko/releases/tag/v0.1.3) (2023-05-13)

### 🐛 Bug Fixes

- **cache:** :bug: Cache modules exposed in subpackage __init__. ([`fd65e9d`](https://github.com/klarna-incubator/mleko/commit/fd65e9df668a0003543a1fa2e3260f723a87285f))

## [v0.1.2](https://github.com/klarna-incubator/mleko/releases/tag/v0.1.2) (2023-05-13)

### 🐛 Bug Fixes

- **cache:** :bug: Fixed LRUCacheMixin eviction test case. ([`ce5bfc1`](https://github.com/klarna-incubator/mleko/commit/ce5bfc17aabaf858778eb42ef40c9f38a9e2ae97))
- :bug: Temporarely disabled failing tests for cache. ([`9c17960`](https://github.com/klarna-incubator/mleko/commit/9c17960185606b6e9d150154bc4b6aac6592a7cc))

### 📝 Documentation

- :memo: Fixed sphinx-autoapi build warnings. ([`040963a`](https://github.com/klarna-incubator/mleko/commit/040963ae3e0bcf871ea80591d9efa43aee66c157))

## [v0.1.0](https://github.com/klarna-incubator/mleko/releases/tag/v0.1.0) (2023-05-12)

### ✨ Features

- **data source:** :sparkles: Add KaggleDataSource to download the dataset from Kaggle by providing a destination directory, owner slug, dataset slug, and necessary API credentials. ([`3fa07b6`](https://github.com/klarna-incubator/mleko/commit/3fa07b633a4347c27b576ea93d38946d721babdf))

### 🐛 Bug Fixes

- **cache:** :bug: Fixed test by not testing it... ([`e3a0ce9`](https://github.com/klarna-incubator/mleko/commit/e3a0ce91da26bf424b199105be164c33b3a6b354))
- **cache:** :bug: Try logging using assert to fix GH issue ([`5e247ec`](https://github.com/klarna-incubator/mleko/commit/5e247ec2de355148b1710a1691404f8cffc4bd31))
- **cache:** :bug: Attempting to fix test case failing in GH actions. ([`4892591`](https://github.com/klarna-incubator/mleko/commit/4892591e9ab6244ec6494d9fc9532eeeb8f12c68))
- **cache:** :bug: LRUCacheMixin now relies on file modification time instead of access time due to system limitations. ([`127d657`](https://github.com/klarna-incubator/mleko/commit/127d6578715f5e852e3638d8a037d072991034d7))
- :bug: Fixed docstrings for private methods in KaggleDataSource and removed xdoctest from build steps ([`bb55cf5`](https://github.com/klarna-incubator/mleko/commit/bb55cf55c149a8fdec3458799e987969174b5207))
