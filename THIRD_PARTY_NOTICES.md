# Third-party data and software notices

The MIT license in this repository applies to the original code and documentation. It does not
relicense third-party datasets. No source corpus is included, and the public repository does not
provide automated download scripts. The committed external evidence consists of derived aggregates,
indices, labels, and short classification outputs; copied benchmark-rule text has been removed.

Users who reproduce the external analyses must obtain their own authorized copies, comply with the
source terms, and place the expected files under the ignored `external/` directory. `make
verify-external` fails when those files are absent.

| Resource | Upstream terms recorded for this release | Public-repository treatment |
|---|---|---|
| Mind2Web and the Mind2Web-SC derivative used by AGrail/SeeAct | Mind2Web states CC BY 4.0 for its dataset; SeeAct identifies OpenRAIL terms for data in its repository | Source records and task text are excluded; only derived results and short model classifications are retained |
| eICU-AC / eICU-derived material | The eICU Collaborative Research Database v2.0 requires credentialing, training, and the PhysioNet Credentialed Health Data License and Data Use Agreement 1.5.0 | No patient or question records are redistributed; reproduction requires the user's own authorized copy |
| BeaverTails | CC BY-NC 4.0 for the dataset family | Dataset records are excluded; only derived aggregate measurements are retained |
| ToxicChat | The referenced Hugging Face dataset card identifies CC BY-NC 4.0 | Conversation text is excluded; only derived aggregate measurements are retained |
| R-Judge | The official repository supplies the dataset and citation but no dataset license was identified during this release review | Dataset records are excluded; users must resolve permitted use before reproducing this lane |
| GitHub Actions | `actions/checkout` v4.2.2 and `actions/setup-python` v5.6.0, each pinned to a full commit SHA in the workflow | Executed only in hosted CI with read-only repository permission |
| Ruff | Version 0.13.1 is pinned in the workflow | Development lint tool only; not a runtime dependency |

The external-result citations and exact methods are in `PAPER.md`. This notice records the release
boundary; it is not legal advice and does not claim that a source's terms permit uses beyond those
stated by that source.
