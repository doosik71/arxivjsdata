# metric-learn: Metric Learning Algorithms in Python

William de Vazelhes, CJ Carey, Yuan Tang, Nathalie Vauquier, Aurélien Bellet

## 🧩 Problem to Solve

기존 머신러닝에서 유클리드(Euclidean), 코사인(Cosine) 등 표준 거리 측정법은 특정 데이터와 작업에 적합하지 않은 경우가 많습니다. 도메인 지식에 기반한 거리 설계는 어렵기 때문에, 데이터로부터 작업에 특화된 거리 측정법을 자동으로 학습하는 것이 필요합니다.

현재 존재하는 거리 학습(Metric Learning) 알고리즘들은 파편화되어 있거나 (Matlab), 범용 머신러닝 라이브러리와 통합된 통일된 API가 부족하며 (R의 dml), 특정 분야(예: 딥 러닝)에만 초점을 맞추고 있습니다 (Python의 pytorch-metric-learning, pyDML). 따라서 `scikit-learn`과 호환되는 통합된 파이썬 패키지의 필요성이 제기됩니다.

## ✨ Key Contributions

- **오픈 소스 파이썬 패키지**: `metric-learn`은 지도 학습(supervised) 및 약한 지도 학습(weakly-supervised) 거리 학습 알고리즘을 구현한 오픈 소스 파이썬 패키지입니다.
- **`scikit-learn` 호환성**: `scikit-learn` API와 완벽하게 호환되는 통일된 인터페이스를 제공하여 교차 검증, 모델 선택 및 다른 `scikit-learn` 추정기와의 파이프라이닝을 용이하게 합니다.
- **다양한 알고리즘 구현**: 지도 학습, 쌍(pair) 학습, 삼중항(triplet) 학습, 사중항(quadruplet) 학습 등 다양한 감독 수준을 지원하는 10가지 인기 있는 거리 학습 알고리즘을 구현합니다.
- **Mahalanobis 거리 학습**: 모든 알고리즘은 특성 공간의 선형 변환 $L$ 또는 Mahalanobis 행렬 $M = L^T L$을 학습하여 Mahalanobis 거리를 배웁니다.
- **품질 및 커뮤니티**: 철저한 테스트 커버리지(97%), PyPI를 통한 MIT 라이선스 배포, `scikit-learn-contrib` 조직에 포함되어 높은 품질과 커뮤니티 참여를 보장합니다.

## 📎 Related Works

- **Matlab 구현**: 많은 거리 학습 알고리즘이 저자들에 의해 Matlab으로 구현되었으나 공통 API가 부족합니다.
- **R 패키지 `dml`**: R에서 여러 거리 학습 알고리즘을 통일된 인터페이스로 구현하지만, 범용 머신러닝 라이브러리와 긴밀하게 통합되지는 않았습니다 (Tang et al., 2018).
- **Python 패키지 `pyDML`**: 주로 완전 지도 학습 및 비지도 학습 알고리즘을 포함합니다 (Suárez et al., 2020).
- **`pytorch-metric-learning`**: PyTorch 프레임워크를 사용하는 딥 거리 학습에 중점을 둡니다.
- **개별 알고리즘**: NCA (Goldberger et al., 2004), LMNN (Weinberger and Saul, 2009), RCA (Shental et al., 2002), LFDA (Sugiyama, 2007), MLKR (Weinberger and Tesauro, 2007), MMC (Xing et al., 2002), ITML (Davis et al., 2007), SDML (Qi et al., 2009), SCML (Shi et al., 2014), LSML (Liu et al., 2012) 등의 기존 연구를 참조하여 구현했습니다.

## 🛠️ Methodology

`metric-learn`은 거리 함수 매개변수를 최적화하는 문제로 거리 학습을 공식화합니다.

- **Mahalanobis 거리 학습**: 현재 구현된 모든 알고리즘은 Mahalanobis 거리를 학습합니다. 두 점 $x$와 $x'$ 사이의 Mahalanobis 거리는 $D_L(x, x') = \sqrt{(Lx - Lx')^T(Lx - Lx')}$로 정의됩니다. 이는 선형 변환 $L$을 적용한 후의 유클리드 거리와 동일하며, $M=L^T L$인 Mahalanobis 행렬 $M$을 사용하여 $D_L(x, x') = \sqrt{(x - x')^T M (x - x')}$으로도 표현될 수 있습니다.
- **다양한 감독 유형**:
  - **지도 학습 (Supervised Learners)**: 각 훈련 예제에 하나의 레이블이 있는 데이터셋에서 학습하여 같은 클래스의 점들은 가깝게, 다른 클래스의 점들은 멀리 떨어뜨립니다 (예: NCA, LMNN, RCA, LFDA, MLKR).
  - **쌍 학습 (Pair Learners)**: 각 쌍이 유사하거나 비유사한지 레이블이 지정된 점 쌍 집합에서 학습하여 유사한 쌍은 가깝게, 비유사한 쌍은 멀리 떨어뜨립니다 (예: MMC, ITML, SDML).
  - **삼중항 학습 (Triplet Learners)**: 3개 점의 튜플에 대해 학습하여 첫 번째 (기준) 점이 두 번째 점에는 더 가깝고 세 번째 점에는 더 멀리 떨어지도록 합니다 (예: SCML).
  - **사중항 학습 (Quadruplet Learners)**: 4개 점의 튜플에 대해 학습하여 첫 번째 두 점이 마지막 두 점보다 더 가깝게 되도록 합니다 (예: LSML).
- **소프트웨어 아키텍처 및 API**:
  - 모든 거리 학습기는 `scikit-learn`의 `BaseEstimator`를 상속받는 `BaseMetricLearner` 추상 클래스로부터 상속받습니다.
  - `get_metric()` (거리 계산 함수 반환) 및 `score_pairs()` 메서드를 구현해야 합니다.
  - Mahalanobis 거리 학습 알고리즘은 변환 행렬 $L$에 해당하는 `components_` 속성을 가진 `MahalanobisMixin` 인터페이스를 상속받습니다.
  - 지도 학습기는 `scikit-learn`의 `TransformerMixin`을 상속받아 `sklearn.pipeline.Pipeline`을 통한 파이프라이닝을 지원합니다.
  - 약한 지도 학습 알고리즘은 3차원 배열 형태의 튜플 (쌍, 삼중항, 사중항)에 대해 `fit` 및 `predict`를 수행합니다.

## 📊 Results

이 논문은 주로 `metric-learn` 패키지의 기능과 API를 설명하며, 새로운 연구 결과를 제시하기보다는 패키지의 유용성을 강조합니다.

- **성공적인 알고리즘 구현**: 10가지 인기 있는 거리 학습 알고리즘이 성공적으로 구현되어 다양한 감독 시나리오를 처리할 수 있습니다.
- **높은 코드 품질**: 97%의 높은 테스트 커버리지를 통해 코드의 신뢰성을 확보했습니다.
- **`scikit-learn` 생태계 통합**: `scikit-learn-contrib` 조직의 일부로 인정받아 `scikit-learn` 생태계 내에서의 호환성과 품질을 입증했습니다.
- **커뮤니티 관심**: GitHub에서 1000개 이상의 별(stars)과 200개 이상의 포크(forks)를 기록하며 활발한 커뮤니티 관심을 받았습니다.
- **쉬운 통합 시연**: `scikit-learn.pipeline.Pipeline` 및 `GridSearchCV`와의 통합 예시를 통해 지도 학습 설정 (예: LMNN + KNeighborsClassifier)에서의 간편한 사용을 보여주었습니다.
- **약한 지도 학습 지원**: `fetch_lfw_pairs` 데이터셋에서 MMC를 사용한 교차 검증 예시를 통해 약한 지도 학습 시나리오에서도 API의 사용 편의성을 시연했습니다.

## 🧠 Insights & Discussion

- **의미**: `metric-learn`은 `scikit-learn`과 완벽하게 호환되는 거리 학습 프레임워크를 제공함으로써 파이썬 머신러닝 생태계의 중요한 공백을 메웠습니다. 이를 통해 강력한 거리 학습 기법을 기존 ML 워크플로우에 더 쉽게 통합하고 접근 가능하게 만들었습니다.
- **영향**: 이 패키지는 최근접 이웃(nearest neighbors) 기반 모델, 클러스터링(clustering), 검색(retrieval) 및 차원 축소(dimensionality reduction) 작업의 성능을 향상시키는 데 기여할 수 있습니다.
- **장점**: 통일된 API, `scikit-learn` 호환성, 다양한 감독 유형 지원, 오픈 소스 및 잘 테스트된 코드 기반은 개발자와 연구자에게 큰 이점입니다.
- **제한 사항 및 향후 계획**:
  - **확장성**: 대규모 데이터셋 처리를 위해 확률적 솔버(SGD 및 변형) 및 즉석 튜플 배치 처리를 구현할 계획입니다.
  - **알고리즘 확장**: 다중 레이블 (Liu and Tsang, 2015) 및 고차원 문제 (Liu and Bellet, 2019)를 처리하는 최신 알고리즘을 포함할 예정입니다.
  - **다양한 거리 형태 지원**: 이선형 유사성(bilinear similarities), 비선형 및 지역 거리(nonlinear and local metrics)와 같은 다른 형태의 거리 학습을 지원할 계획입니다.

## 📌 TL;DR

**문제**: 표준 거리 측정법은 특정 머신러닝 작업에 부적합한 경우가 많으며, 기존 거리 학습 알고리즘 구현은 `scikit-learn`과 호환되는 통합 파이썬 인터페이스가 부족했습니다.
**방법**: `metric-learn`은 지도 학습, 쌍, 삼중항, 사중항 학습 등 다양한 Mahalanobis 거리 학습 알고리즘을 구현한 오픈 소스 파이썬 패키지입니다. 이 패키지는 `scikit-learn`과 완벽하게 호환되는 API를 제공합니다.
**발견**: `metric-learn`은 거리 학습을 머신러닝 파이프라인, 교차 검증, 모델 선택과 쉽게 통합할 수 있는 간소화된 방법을 제공하여 파이썬 ML 생태계의 핵심적인 격차를 해소합니다.
