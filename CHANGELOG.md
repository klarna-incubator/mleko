# Changelog

<!--next-version-placeholder-->

## v0.5.0 (2023-06-17)

### Feature

* **transformer:** ✨ Add MinMaxScalerTransformer for normalizing numerical features. ([`9b26c00`](https://github.com/ErikBavenstrand/mleko/commit/9b26c00ecefa12929a2316309ba5b3e55022b5a5))
* **transformer:** ✨ Add MaxAbsScalerTransformer that scales numerical features. ([`1fd2a93`](https://github.com/ErikBavenstrand/mleko/commit/1fd2a9369bfb74eba3d0015024857070f31bbe34))
* **transformer:** ✨ Add CompositeTransformer for chaining together multiple transformers sequentially. ([`006d741`](https://github.com/ErikBavenstrand/mleko/commit/006d74157167e4ddb2c575ca897662de8a1c0d4d))
* **transformer:** ✨ Add LabelEncoderTransformer for ordinal encoding. ([`41a4c45`](https://github.com/ErikBavenstrand/mleko/commit/41a4c45dcb7a232f2a8f3af3a39e9bcf4bcb3929))
* **transformer:** ✨ Add FrequencyEncoderTransformer along with support for pipeline. ([`465e6db`](https://github.com/ErikBavenstrand/mleko/commit/465e6db3b2830e3cc3f996af9eacb36b8ccf8468))

### Refactor

* 💫 Switch to tqdm.auto to prevent breaking in Jupyter notebooks. ([`dc139cf`](https://github.com/ErikBavenstrand/mleko/commit/dc139cf3844d1a6beb7b5682b44275fd2f9ef2cf))

### Test

* ✅ Now _get_local_filenames returns a sorted list of filenames to ensure stability. ([`774e8eb`](https://github.com/ErikBavenstrand/mleko/commit/774e8eb2a38e5904a170473c56523926b3acffb4))

### Chore

* **deps:** Fix issue with boolean logic in last commit. ([`3832911`](https://github.com/ErikBavenstrand/mleko/commit/38329116ab1da1a19b14e201c283b3b50435e7c0))
* **deps:** Update release workflow to ignore chore(deps) commits. ([`a7d5f62`](https://github.com/ErikBavenstrand/mleko/commit/a7d5f62dba2ef4686c77e5bc41f2e9546d6049ca))
* **deps:** Bump sphinx-autoapi from 2.1.0 to 2.1.1 in /docs ([`2cb82d1`](https://github.com/ErikBavenstrand/mleko/commit/2cb82d1d791ce4f8cb379071250e36506931f8cd))

## v0.4.2 (2023-06-11)

### Performance

* ⚡️ Optimize VarianceFeatureSelector when threshold is 0. ([`906dde3`](https://github.com/ErikBavenstrand/mleko/commit/906dde391d83bc3b07ac5a56bb690dba294e757f))

### Refactor

* ➖ Remove pandas dependency. ([`40e264c`](https://github.com/ErikBavenstrand/mleko/commit/40e264c0604793e860887bbbd03b5c374fa55162))

### Ci

* **semantic versioning:** 👷 Add more sections to changelog based on conventional commit categories. ([`e5b1594`](https://github.com/ErikBavenstrand/mleko/commit/e5b15944ae98793273ed407b13867abf68783f36))

## v0.4.1 (2023-06-04)
### Fix

* **feature selection:** 🐛 Fix `FeatureSelector` cache to use tuple instead of frozenset to have stable fingerprint. ([`cd82417`](https://github.com/ErikBavenstrand/mleko/commit/cd824177e252ae00af4d2ed8cc50b607c4a4928f))

## v0.4.0 (2023-06-03)
### Feature

* **feature selection:** ✨ Add  that filters out invariant features. ([`798c261`](https://github.com/ErikBavenstrand/mleko/commit/798c26103b41fd32f71d16b7b8d40f647f2c849b))
* **feature selection:** ✨ Add `PearsonCorrelationFeatureSelector` which drops highly correlated features. ([`66e5cd2`](https://github.com/ErikBavenstrand/mleko/commit/66e5cd28a4621172ed1c3d27cfc430b89d7da321))
* **feature selection:** ✨ Add `CompositeFeatureSelector`, for chaining multiple feature selection steps on the same DataFrame. ([`3d75079`](https://github.com/ErikBavenstrand/mleko/commit/3d75079d1bc9ab1102a9edb6dbb545b03f43b4dd))
* **feature selection:** ✨ Add standard deviation feature selector. ([`c56177b`](https://github.com/ErikBavenstrand/mleko/commit/c56177bb7e25cbf763418707d09976290f659088))
* **feature selection:** ✨ Add missing rate feature selector. ([`d5ba8b5`](https://github.com/ErikBavenstrand/mleko/commit/d5ba8b57ae4102b2c5c424556d86640f2934dc46))

### Fix

* 🐛 Fix typeguard breaking changes causing build to fail. ([`66c6a8e`](https://github.com/ErikBavenstrand/mleko/commit/66c6a8e24e04815f35dc18de5b3ccd20589fe79d))

## v0.3.1 (2023-05-21)
### Fix
* :bug: Added notes to pipeline step docstrings. ([`d94f899`](https://github.com/ErikBavenstrand/mleko/commit/d94f89921befafe6a81540c996626e7a849fd260))

## v0.3.0 (2023-05-21)
### Feature
* New notes ([#54](https://github.com/ErikBavenstrand/mleko/issues/54)) ([`21239f7`](https://github.com/ErikBavenstrand/mleko/commit/21239f7f7f26cb8fe7b419e81a3e7fe9dd3736fd))

### Fix
* **data splitting:** :bug: Added notes and examples to splitters docstrings. ([`d162c86`](https://github.com/ErikBavenstrand/mleko/commit/d162c8611c0284d0a83e4b5ad9664f964351d194))
* **pipeline:** :bug: Updated some docstrings. ([`56b36fd`](https://github.com/ErikBavenstrand/mleko/commit/56b36fd70030b2d39351b74b3c0e4dce5bd9a06c))

## v0.2.0 (2023-05-21)
### Feature
* Add data splitting step ([#53](https://github.com/ErikBavenstrand/mleko/issues/53)) ([`a668b1a`](https://github.com/ErikBavenstrand/mleko/commit/a668b1a0cd61b6fa1970f401f8209accd40e083f))

### Documentation
* Removed duplicate row. ([`5d77131`](https://github.com/ErikBavenstrand/mleko/commit/5d77131341fb9d241fb179115f1dbd51c9962059))
* Adding pre-commit check for conventional commits. ([`dd2076e`](https://github.com/ErikBavenstrand/mleko/commit/dd2076e38711368a94ba7b454ce830caf23ebf1d))

## v0.1.3 (2023-05-13)
### Fix
* **cache:** :bug: Cache modules exposed in subpackage __init__. ([`fd65e9d`](https://github.com/ErikBavenstrand/mleko/commit/fd65e9df668a0003543a1fa2e3260f723a87285f))

## v0.1.1 (2023-05-13)
### Fix
* :bug: Temporarely disabled failing tests for cache. ([`9c17960`](https://github.com/ErikBavenstrand/mleko/commit/9c17960185606b6e9d150154bc4b6aac6592a7cc))

### Documentation
* :memo: Fixed sphinx-autoapi build warnings. ([`040963a`](https://github.com/ErikBavenstrand/mleko/commit/040963ae3e0bcf871ea80591d9efa43aee66c157))
