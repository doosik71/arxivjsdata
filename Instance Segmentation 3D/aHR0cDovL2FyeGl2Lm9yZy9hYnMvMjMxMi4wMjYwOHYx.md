# panoptica – instance-wise evaluation of 3D semantic and instance segmentation maps

Florian Kofler et al. (2023)

## 🧩 Problem to Solve

생물 의학 영상 분석(Biomedical Image Analysis)에서 분할(Segmentation) 작업은 주로 모든 픽셀/복셀을 클래스별로 구분하는 Semantic Segmentation(SemS)과 개별 객체(Instance)를 구분하는 Instance Segmentation(InS)으로 나뉜다. 많은 임상 상황, 예를 들어 다발성 경화증(Multiple Sclerosis)의 뇌 병변 분석에서는 개별 병변의 개수와 위치를 정확히 파악하는 Instance-wise evaluation이 과학적 연구와 진단 및 모니터링 측면에서 매우 중요하다.

그러나 실제 연구 현장에서는 다음과 같은 문제로 인해 Instance-wise 분석이 충분히 이루어지지 않고 있다.

- 적절한 Instance Label의 부족으로 인해 많은 문제가 SemS 형태로 다루어지고 있다.
- 기존의 Panoptic Quality(PQ) 측정 도구들(MONAI, torchmetrics 등)은 대부분 2D 데이터만을 지원하거나, 3D 구현체라 하더라도 계산 효율성이 떨어지고 Instance Matching 기능이 부족하여 대규모 데이터셋 평가에 부적합하다.
- Semantic Segmentation 결과물을 Instance 수준에서 평가하고 싶어도 이를 위한 체계적인 파이프라인이 부족하다.

따라서 본 논문의 목표는 2D 및 3D 세그멘테이션 맵에서 인스턴스별 품질 지표를 효율적으로 계산할 수 있는 모듈형 오픈소스 패키지인 `panoptica`를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 생물 의학 영상의 특성을 고려하여 설계된 성능 최적화 패키지 `panoptica`의 제안이다. 주요 설계 아이디어는 다음과 같다.

- **모듈형 3단계 파이프라인**: Instance Approximation $\rightarrow$ Instance Matching $\rightarrow$ Instance Evaluation로 이어지는 구조를 통해, Semantic 맵을 Instance 맵으로 변환하고 이를 매칭하여 평가하는 과정을 유연하게 구성하였다.
- **범용적 지표 지원**: 기존의 IoU 기반 PQ뿐만 아니라 Dice Similarity Coefficient(DSC), Average Symmetric Surface Distance(ASSD)와 같은 다양한 유사도 지표를 PQ 프레임워크에 통합하였다.
- **3D 최적화 및 성능 향상**: Python의 `multiprocessing`을 이용한 병렬 계산과, 예측 및 참조 영역의 결합 Bounding Box를 이용한 Cropping 기법을 통해 대규모 3D 데이터셋에서도 빠른 계산이 가능하도록 구현하였다.
- **Semantic-to-Instance 평가**: Semantic Segmentation 결과물을 Connected Component Analysis(CCA)를 통해 인스턴스화함으로써, 기존 SemS 모델들을 Instance 수준에서 평가할 수 있는 환경을 제공한다.

## 📎 Related Works

논문에서는 Panoptic Segmentation의 기초가 되는 Kirillov et al.의 연구와 함께, 기존의 메트릭 구현체들의 한계를 지적한다.

- **Panoptic Quality (PQ)**: Semantic Segmentation과 Instance Segmentation을 통합하여 평가하는 지표로, 탐지 품질(Detection Quality)과 분할 품질(Segmentation Quality)을 동시에 측정할 수 있다.
- **기존 구현체의 한계**:
  - **MONAI & torchmetrics**: 주로 2D 데이터만 지원하며, 특히 `torchmetrics`는 인스턴스 매칭 기능이 없어 이미 매칭된 맵에만 적용 가능하다.
  - **Metrics Reloaded (MONAI side package)**: 3D를 지원하지만, 한 번의 호출에 하나의 지표만 반환하므로 여러 지표를 구할 때 계산 오버헤드가 매우 크며, TP/FP/FN과 같은 세부 정보를 반환하지 않는다.
  - **pymia**: 다양한 지표를 제공하지만, 이미지 전체 단위로 작동하며 인스턴스 근사(Approximation)나 매칭 기능을 제공하지 않는다.

`panoptica`는 이러한 한계를 극복하여 3D 지원, 빠른 계산 속도, 모듈형 매칭 및 근사 기능을 모두 제공함으로써 차별점을 갖는다.

## 🛠️ Methodology

`panoptica`는 다음과 같은 3단계 파이프라인을 통해 지표를 계산한다.

### 1. Instance Approximation (인스턴스 근사)

Semantic Segmentation 맵(클래스 정보만 있는 맵)을 인스턴스 맵으로 변환하는 단계이다.

- **방법**: Connected Component Analysis(CCA)를 사용한다.
- **백엔드**: 2D 데이터에는 `scipy`가, 3D 데이터에는 `cc3d`가 더 효율적이므로 입력 데이터의 차원에 따라 자동으로 백엔드를 선택한다.

### 2. Instance Matching (인스턴스 매칭)

예측된 인스턴스를 참조(Reference) 인스턴스와 연결하는 단계이다. IOU 또는 DSC를 Matching Metric(MM)으로 사용한다.

- **Naive Threshold Matcher**: 사용자가 설정한 임계값(Threshold)을 넘는 인스턴스 중 MS(Matching Score)가 가장 높은 것을 매칭한다.
- **Maximize Many-to-One Matcher**: 여러 개의 예측 인스턴스가 하나의 참조 인스턴스에 매칭될 때, 그 합집합의 MS가 개별 인스턴스보다 높다면 이를 허용하여 전체적인 매칭 점수를 최대화한다.

### 3. Instance Evaluation (인스턴스 평가)

매칭된 결과를 바탕으로 최종 지표를 산출한다. 핵심 지표인 Panoptic Quality(PQ)는 다음과 같이 정의된다.

$$PQ = \frac{\sum_{(R,P) \in TP} f(R,P)}{|TP| + 0.5|FP| + 0.5|FN|}$$

여기서 $f(R,P)$는 유사도 지표(IoU, DSC, ASSD 등)이며, $TP$는 True Positive, $FP$는 False Positive, $FN$은 False Negative 인스턴스의 집합이다. PQ는 다음과 같이 분해하여 해석할 수 있다.

$$PQ = \underbrace{\frac{\sum_{(R,P) \in TP} f(R,P)}{|TP|}}_{\text{Segmentation Quality (SQ)}} \cdot \underbrace{\frac{|TP|}{|TP| + 0.5|FP| + 0.5|FN|}}_{\text{Recognition Quality (RQ)}}$$

- **SQ (Segmentation Quality)**: 올바르게 탐지된 인스턴스들이 얼마나 정확하게 분할되었는지를 측정한다.
- **RQ (Recognition Quality)**: 인스턴스들을 얼마나 정확하게 탐지(Detection)했는지를 측정하며, F1-score와 동등하다.

## 📊 Results

본 연구에서는 세 가지 생물 의학 데이터셋을 통해 `panoptica`의 유효성을 검증하였다.

### 1. VerSe (척추 분할 실험)

- **목적**: 다중 클래스 인스턴스 분할 성능 평가.
- **결과**: gvDSC(Global volumetric Dice)만으로는 알 수 없는 세부 성능을 발견하였다. 예를 들어, 특정 모델은 gvDSC는 가장 높았으나 SQ, PQ, SQASSD에서는 다른 모델보다 낮게 나타났다. 이는 해당 모델이 인스턴스 탐지(RQ)는 뛰어나지만, 탐지된 인스턴스의 경계 분할(SQ) 능력은 상대적으로 낮음을 의미한다.

### 2. ISLES (뇌졸중 병변 분할 실험)

- **목적**: Semantic Segmentation 맵을 Instance-wise로 평가하는 사례 제시.
- **결과**: 기존 챌린지에서는 gvDSC 위주로 평가했으나, `panoptica`를 통해 분석한 결과 gvDSC 성능이 가장 좋은 모델이 반드시 인스턴스 수준(PQ, SQ)에서 최선의 성능을 보이는 것은 아님을 확인하였다.

### 3. BraTS Mets (전이성 뇌종양 분할 실험)

- **목적**: Glioma(교종) 전용 알고리즘이 Metastasis(전이암) 분할에 얼마나 일반화되는지 평가.
- **결과**: 모든 알고리즘에서 RQ가 매우 낮게 나타났으며, 이는 모델들이 수많은 작은 전이성 병변들을 제대로 탐지하지 못하고 있음을 보여준다. gvDSC 수치보다 PQ 수치가 훨씬 낮게 나타나, 단순 부피 기반 지표가 실제 임상적 탐지 실패를 은폐할 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 의의

`panoptica`는 단순한 지표 계산기를 넘어, 생물 의학 영상 분석에서 간과되었던 **'인스턴스 수준의 분석'**을 표준화된 파이프라인으로 제공한다. 특히 Semantic 맵을 Instance 맵으로 변환하여 평가할 수 있게 함으로써, 기존의 수많은 SemS 모델들을 재평가할 수 있는 도구를 제공했다는 점이 높게 평가된다.

### 한계 및 향후 과제

- **CCA의 한계**: Semantic 맵을 인스턴스화할 때 CCA를 사용하면 픽셀 단위의 클래스 확률(Soft-score) 정보가 소실되고 정수 레이블만 남게 된다.
- **지표 확장 필요성**: 현재는 IoU, DSC, ASSD 위주이나, 향후 clDice, NSD, Hausdorff Distance 등 경계 기반 지표 및 Average Precision(AP)과 같은 다중 임계값 지표의 추가가 필요하다.
- **매칭 알고리즘 고도화**: 현재의 단순 매칭 외에 Hungarian Matching과 같은 최적 매칭 알고리즘이나 위상적 특징(Topological features) 기반의 매칭 전략 도입이 논의된다.

## 📌 TL;DR

본 논문은 2D 및 3D 생물 의학 영상의 인스턴스별 분할 품질을 정밀하게 측정하기 위한 오픈소스 파이프라인 `panoptica`를 제안한다. 이 도구는 **[근사 $\rightarrow$ 매칭 $\rightarrow$ 평가]**의 모듈형 구조를 가지며, 특히 Semantic Segmentation 결과를 인스턴스 수준에서 분석할 수 있게 하여, 단순 부피 기반 지표(gvDSC)가 놓치기 쉬운 탐지 오류와 분할 정밀도를 분리하여 평가(RQ vs SQ)할 수 있게 한다. 이는 향후 정밀한 병변 분석이 필요한 의료 AI 모델의 평가 표준을 높이는 데 기여할 것으로 보인다.
