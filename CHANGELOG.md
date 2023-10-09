# Changelog

<!--next-version-placeholder-->

## v1.2.0 (2023-10-09)

### Feature

- **data source:** ✨ Add support for pattern matching in `*Ingester` and add `LocalManifest` to index fetched files. ([`75974a4`](https://github.com/klarna-incubator/mleko/commit/75974a40aa1af5e21a4dedc7e0d05be2153ec7aa))

### Fix

- **logging:** 🐛 Fix LGBM logging routing to correct log level. ([`0e5fa77`](https://github.com/klarna-incubator/mleko/commit/0e5fa77ae14ccd27eb9b435ec8a73815fe764350))

### Style

- Remove unnecessary blank lines ([`a06edf2`](https://github.com/klarna-incubator/mleko/commit/a06edf23e865e6bfb95f9f3d13611a24de12b4ba))
- ✏️ Improve logging of `CSVToVaexConverter` and fix typo in `write_vaex_dataframe`. ([`197e56a`](https://github.com/klarna-incubator/mleko/commit/197e56ad4af59da2a08762c70b8e7ae5c5e350e2))

### Build

- 🔒️ Bump `gitpython` to resolve CVE-2023-41040 and CVE-2023-40590. ([`79627bd`](https://github.com/klarna-incubator/mleko/commit/79627bda5228d0470b2a3ca4613ae7636d79f4b6))

### Chore

- Update ignored files, add pattern for dev notebooks and .aider files ([`fff4afb`](https://github.com/klarna-incubator/mleko/commit/fff4afb8b279534b6ee7c5dfc1fb1d6608606a74))
- **deps:** Bump sphinx-autoapi from 2.1.1 to 3.0.0 in /docs ([#96](https://github.com/klarna-incubator/mleko/issues/96)) ([`f2f9194`](https://github.com/klarna-incubator/mleko/commit/f2f91947f5300d5cb5511ede607e586fdae80abe))
- **deps:** Bump sphinx-autoapi from 2.1.1 to 3.0.0 in /docs ([`d0e5237`](https://github.com/klarna-incubator/mleko/commit/d0e5237698c05eba13ba9b203ee133e9491bc4fe))

## v1.1.0 (2023-09-27)

### Feature

- **tuning:** ✨ Add hyperparameter tuning functionality, initially including `OptunaTuner`. ([`be38c07`](https://github.com/klarna-incubator/mleko/commit/be38c075e269ed2e42560dfa26ac71ab08448444))

### Test

- **tuning:** 🧪 Add test cases for `TuneStep`. ([`d811c7d`](https://github.com/klarna-incubator/mleko/commit/d811c7d4f60b4440b669cf723d3b85ee962efc97))

## v1.0.0 (2023-09-20)

### Feature

- **transformer:** ✨ Add `DataSchema` API to transformers `fit`, `transform` and `fit_transform`. ([`e053c85`](https://github.com/klarna-incubator/mleko/commit/e053c854238614ef23c6e8ff7eedc004cbeedf71))

### Breaking

- Forcefully trigger major release. ([`b388b59`](https://github.com/klarna-incubator/mleko/commit/b388b59a9c8da0df56da532a0cc17b347089a146))

### Documentation

- 📝 Improve `README.md` with more up to date information. ([`b388b59`](https://github.com/klarna-incubator/mleko/commit/b388b59a9c8da0df56da532a0cc17b347089a146))
- 📝 Add example notebook for `Titanic` dataset. ([`e651af9`](https://github.com/klarna-incubator/mleko/commit/e651af99d96f2d012faaf6ab5c73cc681f7a5b5b))

### Chore

- **deps:** Bump virtualenv from 20.24.4 to 20.24.5 in /.github/workflows ([#94](https://github.com/klarna-incubator/mleko/issues/94)) ([`eb7f481`](https://github.com/klarna-incubator/mleko/commit/eb7f4816c2c1d574816b05020da16650591ee19e))
- **deps:** Bump crazy-max/ghaction-github-labeler from 4.1.0 to 5.0.0 ([#93](https://github.com/klarna-incubator/mleko/issues/93)) ([`d49f4ac`](https://github.com/klarna-incubator/mleko/commit/d49f4ac5e573ad85f5b98d2a13e14b63f9a2cdf7))
- **deps:** Bump virtualenv in /.github/workflows ([`ab905be`](https://github.com/klarna-incubator/mleko/commit/ab905be4d75e13f10fe62b71f57ef49266ee4ebe))
- **deps:** Bump crazy-max/ghaction-github-labeler from 4.1.0 to 5.0.0 ([`ae40b84`](https://github.com/klarna-incubator/mleko/commit/ae40b8454703eca380b063c6c6851796dc5bcb53))

## v0.8.1 (2023-09-07)

### Fix

- **config:** 🐛 Fix readthedocs build to only generate html. ([`13fc207`](https://github.com/klarna-incubator/mleko/commit/13fc207adbba82fa9cc2d4dcad91038aecafa34f))

### Chore

- **deps:** Bump virtualenv from 20.22.0 to 20.24.4 in /.github/workflows ([#83](https://github.com/klarna-incubator/mleko/issues/83)) ([`67540db`](https://github.com/klarna-incubator/mleko/commit/67540db6689ff184641c829eea81fd1481c8c457))
- **deps:** Bump virtualenv in /.github/workflows ([`c56b133`](https://github.com/klarna-incubator/mleko/commit/c56b1334895dcb6e17641084b0abdb9e8accfc1c))
- **deps:** Bump pip from 23.2 to 23.2.1 in /.github/workflows ([#82](https://github.com/klarna-incubator/mleko/issues/82)) ([`f4b6b0d`](https://github.com/klarna-incubator/mleko/commit/f4b6b0d1ec85c9f07171ff90ab7b22a73d10ee4f))
- **deps:** Bump pip from 23.2 to 23.2.1 in /.github/workflows ([`96d5b3d`](https://github.com/klarna-incubator/mleko/commit/96d5b3db1bcacdea95032ef97f3b8f698e599156))

## v0.8.0 (2023-09-06)

### Feature

- **model:** ✨ Add `LGBMModel` along with base class which can be extended for all types of future models. ([`b47a241`](https://github.com/klarna-incubator/mleko/commit/b47a2412e8925226d5612e4b57c521f562c51636))
- ✨ Add` DataSchema` which tracks dataset features throughout the pipeline and methods. ([`e03bd2c`](https://github.com/klarna-incubator/mleko/commit/e03bd2cbc9e75347eeea37f5f818d05a2548c3de))
- **feature selection:** ✨ Update `BaseFeatureSelector` and children to use the `fit`, `transform` and `fit_transform` pattern. ([`62e4dd1`](https://github.com/klarna-incubator/mleko/commit/62e4dd13a37a852f239c490d2693b20665095cf6))
- **transformer:** ✨ Add `fit`, `transform` and `fit_transform` to all `Transformers`, along with API and caching simplificatons. ([`5cc4ebc`](https://github.com/klarna-incubator/mleko/commit/5cc4ebc9da3b3f3aa5e8788d1237ac2c79ee4693))
- **cache:** ✨ Add `CacheHandler` which allows customization of read/write functions for each cached return value individually. ([`609e084`](https://github.com/klarna-incubator/mleko/commit/609e084e7c354bed5a298b3a3a1c3b033d3db71d))

### Fix

- **feature selection:** 🐛 Add `DataSchema` as partial return from all `fit` methods in feature selectors. ([`ebf2484`](https://github.com/klarna-incubator/mleko/commit/ebf2484fbaac7bbe067ab1491964bc50f4d7d711))

### Refactor

- **cache:** 🚸 Replace `disable_cache` with a check if `cache_size=0` for `LRUCacheMixin`. ([`cfd7592`](https://github.com/klarna-incubator/mleko/commit/cfd759297d2d0e3b21f0f79827baab7c8b882784))

### Chore

- **deps:** Bump pypa/gh-action-pypi-publish from 1.8.8 to 1.8.10 ([#76](https://github.com/klarna-incubator/mleko/issues/76)) ([`d6dc41f`](https://github.com/klarna-incubator/mleko/commit/d6dc41ff9f92a23dedbdc6be987cdfd00e8dc287))
- **deps:** Bump pypa/gh-action-pypi-publish from 1.8.8 to 1.8.10 ([`a7363ee`](https://github.com/klarna-incubator/mleko/commit/a7363ee91342e7f708439e89064adfd7822f39cc))
- **deps:** Bump sphinx-autodoc-typehints from 1.23.3 to 1.24.0 in /docs ([#75](https://github.com/klarna-incubator/mleko/issues/75)) ([`beece1c`](https://github.com/klarna-incubator/mleko/commit/beece1ca454de60918e3266110b2bdba10b82141))
- **deps:** Bump sphinx-autodoc-typehints in /docs ([`aea624c`](https://github.com/klarna-incubator/mleko/commit/aea624c6cc545f3cc7d7fca2087b924f378d647a))
- **deps:** Bump pip from 23.2 to 23.2.1 in /.github/workflows ([#74](https://github.com/klarna-incubator/mleko/issues/74)) ([`b86840f`](https://github.com/klarna-incubator/mleko/commit/b86840fbe61577b6737498f7331d049a74f84353))
- **deps:** Bump pip from 23.2 to 23.2.1 in /.github/workflows ([`35f6828`](https://github.com/klarna-incubator/mleko/commit/35f6828f7ac58197ac171fb1b2860ef86e98a79f))
- **deps:** Bump pip from 22.3.1 to 23.2 in /.github/workflows ([#73](https://github.com/klarna-incubator/mleko/issues/73)) ([`a0083bd`](https://github.com/klarna-incubator/mleko/commit/a0083bd85a8a880c5f09020c1ea7505cc7e2165a))
- **deps:** Bump pip from 22.3.1 to 23.2 in /.github/workflows ([`de871ea`](https://github.com/klarna-incubator/mleko/commit/de871eaab0ea261cb9b9d6318e8fb295985d00d7))
- **deps:** Bump pypa/gh-action-pypi-publish from 1.8.7 to 1.8.8 ([#72](https://github.com/klarna-incubator/mleko/issues/72)) ([`144a702`](https://github.com/klarna-incubator/mleko/commit/144a70228e6183b906f897454f8ac0b381132e37))
- **deps:** Bump pypa/gh-action-pypi-publish from 1.8.7 to 1.8.8 ([`2747ead`](https://github.com/klarna-incubator/mleko/commit/2747ead5edd089711fa458c12608b01c8dddbc8a))

## v0.7.0 (2023-07-11)

### Feature

- ✨ Add fit transform support to all `FeatureSelector` along with refactoring the `LRUCacheMixin`. ([`3df0601`](https://github.com/klarna-incubator/mleko/commit/3df06011085a1230c00250260374e5fd5d325fad))
- ✨ Add support for separate fitting and transforming inside the pipeline. ([`bb9b7a4`](https://github.com/klarna-incubator/mleko/commit/bb9b7a4ea9920d5588972a29e390bac4017b45af))

### Fix

- **data cleaning:** 🐛 Switched to HDF5 as file format for faster I/O and better SageMaker support. ([`61f9e42`](https://github.com/klarna-incubator/mleko/commit/61f9e42ef215485b16d83be9726040b829d5a3e7))

### Chore

- **deps:** Bump nox-poetry from 1.0.2 to 1.0.3 in /.github/workflows ([#70](https://github.com/klarna-incubator/mleko/issues/70)) ([`5aa535d`](https://github.com/klarna-incubator/mleko/commit/5aa535d3f3e3f3999d8bfce3409bd993ff500af2))
- **deps:** Bump nox-poetry from 1.0.2 to 1.0.3 in /.github/workflows ([`b55b7e2`](https://github.com/klarna-incubator/mleko/commit/b55b7e2b6aa6c88c2afec1711c37ff7d31345a4e))

## v0.6.1 (2023-06-30)

### Fix

- **data cleaning:** 🐛 Fix date32/64[day] not converted to datetime. ([`98f4b26`](https://github.com/klarna-incubator/mleko/commit/98f4b26ef5eab26e47cdcc1fd0f31928e063d49a))
- **data source:** 🐛 Fix bug where S3 buckets with no manifest caused crash. ([`9078845`](https://github.com/klarna-incubator/mleko/commit/90788454dc8b2c1665f79746ae1392a8d4d067dd))

### Build

- **config:** 🔧 Switch mypy for pyright and update configuration. ([`5631aed`](https://github.com/klarna-incubator/mleko/commit/5631aedd059bdd6aed2c8e78d09629d8894e5cb8))

### Chore

- **deps:** Bump pypa/gh-action-pypi-publish from 1.5.0 to 1.8.7 ([#67](https://github.com/klarna-incubator/mleko/issues/67)) ([`74315f4`](https://github.com/klarna-incubator/mleko/commit/74315f47f8c31d956ab8040d7688eb88337f6663))
- **deps:** Bump sphinx-autodoc-typehints from 1.23.2 to 1.23.3 in /docs ([#68](https://github.com/klarna-incubator/mleko/issues/68)) ([`54e0845`](https://github.com/klarna-incubator/mleko/commit/54e08455cdf408a340077eaefdfa0cbedf973e74))
- **deps:** Bump sphinx-autodoc-typehints in /docs ([`5c285be`](https://github.com/klarna-incubator/mleko/commit/5c285beb7b462c4396380be0c8c17547f21e7d16))
- **deps:** Bump pypa/gh-action-pypi-publish from 1.5.0 to 1.8.7 ([`b8f0f29`](https://github.com/klarna-incubator/mleko/commit/b8f0f29a50c05534c314787dc0b87552d6e25460))

## v0.6.0 (2023-06-26)

### Feature

- **cache:** ✨ Add cache_group that can segment an instance cache into different isolated parts. ([`b5c3de5`](https://github.com/klarna-incubator/mleko/commit/b5c3de5f1397bfb9422590ba037f431c2eaad9ac))

### Chore

- **deps:** Bump sphinx-autodoc-typehints in /docs ([`de2e720`](https://github.com/klarna-incubator/mleko/commit/de2e720515b362f65078b31f545842be997152e0))

## v0.5.0 (2023-06-17)

### Feature

- **transformer:** ✨ Add MinMaxScalerTransformer for normalizing numerical features. ([`9b26c00`](https://github.com/klarna-incubator/mleko/commit/9b26c00ecefa12929a2316309ba5b3e55022b5a5))
- **transformer:** ✨ Add MaxAbsScalerTransformer that scales numerical features. ([`1fd2a93`](https://github.com/klarna-incubator/mleko/commit/1fd2a9369bfb74eba3d0015024857070f31bbe34))
- **transformer:** ✨ Add CompositeTransformer for chaining together multiple transformers sequentially. ([`006d741`](https://github.com/klarna-incubator/mleko/commit/006d74157167e4ddb2c575ca897662de8a1c0d4d))
- **transformer:** ✨ Add LabelEncoderTransformer for ordinal encoding. ([`41a4c45`](https://github.com/klarna-incubator/mleko/commit/41a4c45dcb7a232f2a8f3af3a39e9bcf4bcb3929))
- **transformer:** ✨ Add FrequencyEncoderTransformer along with support for pipeline. ([`465e6db`](https://github.com/klarna-incubator/mleko/commit/465e6db3b2830e3cc3f996af9eacb36b8ccf8468))

### Refactor

- 💫 Switch to tqdm.auto to prevent breaking in Jupyter notebooks. ([`dc139cf`](https://github.com/klarna-incubator/mleko/commit/dc139cf3844d1a6beb7b5682b44275fd2f9ef2cf))

### Test

- ✅ Now \_get_local_filenames returns a sorted list of filenames to ensure stability. ([`774e8eb`](https://github.com/klarna-incubator/mleko/commit/774e8eb2a38e5904a170473c56523926b3acffb4))

### Chore

- **deps:** Fix issue with boolean logic in last commit. ([`3832911`](https://github.com/klarna-incubator/mleko/commit/38329116ab1da1a19b14e201c283b3b50435e7c0))
- **deps:** Update release workflow to ignore chore(deps) commits. ([`a7d5f62`](https://github.com/klarna-incubator/mleko/commit/a7d5f62dba2ef4686c77e5bc41f2e9546d6049ca))
- **deps:** Bump sphinx-autoapi from 2.1.0 to 2.1.1 in /docs ([`2cb82d1`](https://github.com/klarna-incubator/mleko/commit/2cb82d1d791ce4f8cb379071250e36506931f8cd))

## v0.4.2 (2023-06-11)

### Performance

- ⚡️ Optimize VarianceFeatureSelector when threshold is 0. ([`906dde3`](https://github.com/klarna-incubator/mleko/commit/906dde391d83bc3b07ac5a56bb690dba294e757f))

### Refactor

- ➖ Remove pandas dependency. ([`40e264c`](https://github.com/klarna-incubator/mleko/commit/40e264c0604793e860887bbbd03b5c374fa55162))

### Ci

- **semantic versioning:** 👷 Add more sections to changelog based on conventional commit categories. ([`e5b1594`](https://github.com/klarna-incubator/mleko/commit/e5b15944ae98793273ed407b13867abf68783f36))

## v0.4.1 (2023-06-04)

### Fix

- **feature selection:** 🐛 Fix `FeatureSelector` cache to use tuple instead of frozenset to have stable fingerprint. ([`cd82417`](https://github.com/klarna-incubator/mleko/commit/cd824177e252ae00af4d2ed8cc50b607c4a4928f))

## v0.4.0 (2023-06-03)

### Feature

- **feature selection:** ✨ Add that filters out invariant features. ([`798c261`](https://github.com/klarna-incubator/mleko/commit/798c26103b41fd32f71d16b7b8d40f647f2c849b))
- **feature selection:** ✨ Add `PearsonCorrelationFeatureSelector` which drops highly correlated features. ([`66e5cd2`](https://github.com/klarna-incubator/mleko/commit/66e5cd28a4621172ed1c3d27cfc430b89d7da321))
- **feature selection:** ✨ Add `CompositeFeatureSelector`, for chaining multiple feature selection steps on the same DataFrame. ([`3d75079`](https://github.com/klarna-incubator/mleko/commit/3d75079d1bc9ab1102a9edb6dbb545b03f43b4dd))
- **feature selection:** ✨ Add standard deviation feature selector. ([`c56177b`](https://github.com/klarna-incubator/mleko/commit/c56177bb7e25cbf763418707d09976290f659088))
- **feature selection:** ✨ Add missing rate feature selector. ([`d5ba8b5`](https://github.com/klarna-incubator/mleko/commit/d5ba8b57ae4102b2c5c424556d86640f2934dc46))

### Fix

- 🐛 Fix typeguard breaking changes causing build to fail. ([`66c6a8e`](https://github.com/klarna-incubator/mleko/commit/66c6a8e24e04815f35dc18de5b3ccd20589fe79d))

## v0.3.1 (2023-05-21)

### Fix

- :bug: Added notes to pipeline step docstrings. ([`d94f899`](https://github.com/klarna-incubator/mleko/commit/d94f89921befafe6a81540c996626e7a849fd260))

## v0.3.0 (2023-05-21)

### Feature

- New notes ([#54](https://github.com/klarna-incubator/mleko/issues/54)) ([`21239f7`](https://github.com/klarna-incubator/mleko/commit/21239f7f7f26cb8fe7b419e81a3e7fe9dd3736fd))

### Fix

- **data splitting:** :bug: Added notes and examples to splitters docstrings. ([`d162c86`](https://github.com/klarna-incubator/mleko/commit/d162c8611c0284d0a83e4b5ad9664f964351d194))
- **pipeline:** :bug: Updated some docstrings. ([`56b36fd`](https://github.com/klarna-incubator/mleko/commit/56b36fd70030b2d39351b74b3c0e4dce5bd9a06c))

## v0.2.0 (2023-05-21)

### Feature

- Add data splitting step ([#53](https://github.com/klarna-incubator/mleko/issues/53)) ([`a668b1a`](https://github.com/klarna-incubator/mleko/commit/a668b1a0cd61b6fa1970f401f8209accd40e083f))

### Documentation

- Removed duplicate row. ([`5d77131`](https://github.com/klarna-incubator/mleko/commit/5d77131341fb9d241fb179115f1dbd51c9962059))
- Adding pre-commit check for conventional commits. ([`dd2076e`](https://github.com/klarna-incubator/mleko/commit/dd2076e38711368a94ba7b454ce830caf23ebf1d))

## v0.1.3 (2023-05-13)

### Fix

- **cache:** :bug: Cache modules exposed in subpackage **init**. ([`fd65e9d`](https://github.com/klarna-incubator/mleko/commit/fd65e9df668a0003543a1fa2e3260f723a87285f))

## v0.1.1 (2023-05-13)

### Fix

- :bug: Temporarely disabled failing tests for cache. ([`9c17960`](https://github.com/klarna-incubator/mleko/commit/9c17960185606b6e9d150154bc4b6aac6592a7cc))

### Documentation

- :memo: Fixed sphinx-autoapi build warnings. ([`040963a`](https://github.com/klarna-incubator/mleko/commit/040963ae3e0bcf871ea80591d9efa43aee66c157))
