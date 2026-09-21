# Attribution and source

ReVeMap is extracted from the `developnew` branch of
[Aidenwu0209/SGF-SGAligner](https://github.com/Aidenwu0209/SGF-SGAligner),
at commit `18ddeca303bf9937e855d3ac3113a90dd852a462`.
Its initial independent commit is an extraction, not a claim of newly collected
experiments or a published paper. The original history remains in that repository.

The original MIT copyright notice for Sayan Deb Sarkar is retained in
[LICENSE](LICENSE). ReVeMap's new packaging, entry points and project documentation
are also distributed under MIT. See [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json)
for file-level source hashes and extracted function origins.

The active pipeline uses measured geometry/observation association. The old SGF
neural inference, learned SGAligner matching/training and GeoTransformer submodule
are not included or required. Some historical module names, comments, configuration
IDs and experiment reports remain for reproducibility and attribution.

DROID-W, SAM 3, individual vision-language model checkpoints, the Orbbec SDK and
their native dependencies are external components. Their code, weights and data
are not redistributed here; their own licenses and access conditions still apply.
Historical experiment reports are preserved as records of the source version.
