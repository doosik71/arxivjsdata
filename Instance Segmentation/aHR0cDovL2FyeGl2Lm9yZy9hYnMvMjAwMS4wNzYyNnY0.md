# PatchPerPix for Instance Segmentation

Peter Hirsch, Lisa Mais, Dagmar Kainmueller (2022)

## 🧩 Problem to Solve

본 논문은 특히 생물 의학 이미지(Biomedical Images) 분야에서 발생하는 복잡한 형태의 인스턴스 분할(Instance Segmentation) 문제를 해결하고자 한다. 기존의 Proposal-based 방식(예: Mask R-CNN)은 객체의 위치와 크기를 Bounding Box로 근사할 수 있다는 가정에 기반한다. 하지만 생물 의학 데이터에서는 객체가 이미지의 넓은 영역에 걸쳐 길게 뻗어 있거나, 여러 객체가 매우 밀집되어 있으며, 서로 겹치거나 교차(Crossover)하는 경우가 빈번하여 Bounding Box만으로는 객체를 효과적으로 구분하기 어렵다.

또한, Proposal-free 방식인 Metric Learning이나 Affinity-based 방법들은 픽셀 단위의 예측에 의존하므로 인스턴스의 전체적인 형상(Shape) 정보를 명시적으로 포착하지 못하며, 겹쳐진 인스턴스를 분리하는 데 한계가 있다. 따라서 본 연구의 목표는 복잡한 형상과 밀집된 클러스터, 교차 영역이 존재하는 환경에서도 효과적으로 인스턴스를 분할할 수 있는 새로운 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지를 한 번의 패스로 처리하면서, 학습된 국소적 형상 패치(Learned Shape Patches)들을 조립하여 전체 인스턴스를 복원하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Dense Local Shape Descriptor 예측**: 각 픽셀에 대해 해당 픽셀이 속한 인스턴스의 국소적 형상 패치를 밀집하게 예측한다.
2. **최적 패치 선택 및 조립**: 예측된 수많은 패치 중 합의(Consensus) 기반의 점수가 높은 패치들을 선택하여 이미지 전경을 덮고, 이들 간의 관계를 그래프로 구축하여 인스턴스를 완성한다.
3. **비반복적 일괄 처리**: 기존의 반복적(Iterative) 확장 방식과 달리, 모든 인스턴스를 단 한 번의 파이프라인(One-pass)을 통해 동시에 분할한다.
4. **다양한 도메인에서의 SOTA 달성**: ISBI 2012 EM segmentation, BBBC010 C. elegans, 그리고 2D/3D 세포핵 데이터셋에서 기존의 최신 기술(SOTA)을 경신하였다.

## 📎 Related Works

본 논문은 기존의 인스턴스 분할 접근 방식들을 다음과 같이 분석하고 차별점을 제시한다.

- **Proposal-based Methods (R-CNN 계열)**: Bounding Box 기반의 접근법은 객체가 크고 복잡하며 서로 겹쳐 있는 생물 의학 이미지에서는 효율성이 떨어진다.
- **Proposal-free Methods (Metric Learning, Affinity-based, Watershed)**: 픽셀 간의 유사도나 어피니티를 학습하지만, 인스턴스의 기하학적 형상을 명시적으로 학습하지 않으므로 겹친 객체 분리에 취약하다.
- **Singling Out Networks (SON)**: 알려진 인스턴스 딕셔너리를 사용하여 객체를 분리하지만, 딕셔너리에 없는 형태의 가변적인 객체를 처리하는 데 한계가 있으며, 패치가 아닌 전체 인스턴스 형태만을 예측하므로 크기가 큰 객체 처리가 어렵다.
- **Iterative Methods (Flood Filling Networks)**: 인스턴스를 하나씩 순차적으로 확장하여 분할하지만, 본 논문의 방법론은 모든 인스턴스를 동시에 처리하여 효율성을 높였다.

## 🛠️ Methodology

### 전체 파이프라인

PatchPerPix의 전체 과정은 **[CNN을 통한 패치 예측] $\rightarrow$ [합의 기반 패치 선택] $\rightarrow$ [패치 어피니티 그래프 구축] $\rightarrow$ [그래프 파티셔닝을 통한 인스턴스 확정]** 순으로 진행된다.

### 1. 국소 형상 패치 예측 (Shape Patch Prediction)

CNN은 입력 이미지 $I$의 각 픽셀 $x$에 대해, 주변 오프셋 세트 $P$에 속하는 픽셀 $x+dx$가 $x$와 동일한 인스턴스에 속하는지 여부를 확률값 $p(x, dx) \in [0, 1]$로 예측한다. 이는 각 픽셀마다 하나의 작은 이진 마스크(Binary Mask) 형태의 패치를 생성하는 것과 같다.

### 2. 인스턴스 조립 (Instance Assembly)

예측된 패치들을 이용하여 다음과 같은 단계로 인스턴스를 조립한다.

**A. 합의 어피니티(Consensus Affinity) 계산**
두 픽셀 $y$와 $z$가 동일한 인스턴스에 속하는지에 대한 합의도를 계산한다.
$$\text{aff}(y,z) := \frac{1}{Z_{\text{aff}}(y,z)} \cdot \left( \sum_{x: \{y,z\} \subset \text{fg}(p_x)} p_x(y)p_x(z) - \sum_{x: y \in \text{fg}(p_x), z \in \text{bg}(p_x)} p_x(y)(1-p_x(z)) - \dots \right)$$
여기서 $Z_{\text{aff}}(y,z)$는 $y$와 $z$를 모두 포함하는 패치들의 수이며, 전경($\text{fg}$)과 배경($\text{bg}$) 예측값을 종합하여 두 픽셀의 연결 강도를 결정한다.

**B. 패치 점수 계산 및 선택**
각 패치 $p_x$가 위에서 계산된 합의 어피니티와 얼마나 일치하는지를 측정하여 점수를 매긴다.
$$\text{score}(p_x) := \frac{1}{Z_{\text{score}}(p_x)} \cdot \left( \sum_{\{y,z\} \subset \text{fg}(p_x)} \text{aff}(y,z) - \sum_{y \in \text{fg}(p_x), z \in \text{bg}(p_x)} \text{aff}(y,z) \right)$$
이후 Greedy Set Cover 알고리즘을 사용하여 점수가 높은 패치들을 우선적으로 선택함으로써 이미지의 전경 영역을 모두 덮도록 한다.

**C. 패치 어피니티 그래프 및 파티셔닝**
선택된 패치들 사이의 어피니티 $\text{paff}(p_x, p_y)$를 계산하여 그래프를 생성한다. 이 그래프를 Connected Component Analysis(CC) 또는 Mutex Watershed(MWS) 알고리즘으로 파티셔닝하여 최종 인스턴스 ID를 부여한다.

### 3. CNN 아키텍처

본 논문은 U-Net을 백본으로 사용하며, 세 가지 변형 모델을 제안한다.

- **ppp**: U-Net이 직접 패치($p_x$)를 출력하는 기본 모델.
- **ppp+ae**: 미리 학습된 Autoencoder의 Decoder를 사용하여 압축된 코드(Encoding)를 패치로 복원하는 방식.
- **ppp+dec**: U-Net과 Decoder를 연결하여 End-to-End로 함께 학습시키는 방식. 실험 결과, 이 방식이 가장 우수한 성능을 보였다.

### 4. 겹침 영역(Overlapping Regions) 처리

객체가 겹치는 영역은 형상 패치가 불분명하므로, 네트워크가 별도로 'Overlap' 클래스를 예측하게 하여 해당 영역의 픽셀들을 조립 파이프라인에서 제외함으로써 오분류를 방지한다.

## 📊 Results

### 실험 설정

- **데이터셋**: BBBC010 (C. elegans), ISBI 2012 (EM neurons), dsb2018 (2D nuclei), nuclei3d (3D nuclei).
- **평가 지표**: $\text{avS}$ (Average Score), rRAND, rINF, AP 등.

### 주요 결과

1. **BBBC010 (Worm)**: SON, Harmonic Embeddings 등 기존 방법론보다 월등한 성능을 보였으며, 특히 `ppp+dec` 모델이 가장 높은 정확도를 기록하였다.
2. **ISBI 2012 (EM Neurons)**: 리더보드 기준 rRAND 점수에서 SOTA를 달성하였다. 합의 기반의 패치 선택 과정이 개별 픽셀의 오류를 수정하여 성능을 높였음을 확인하였다.
3. **Nuclei (2D/3D)**: StarDist와 비교했을 때, 단순한 타원형 객체에 대해서도 매우 높은 픽셀 정확도(High IoU)를 보여주었다. 다만 3D 데이터의 경우 계산 복잡도 문제로 패치 크기를 작게 설정해야 했기에 성능 향상 폭이 2D보다 적었다.
4. **3D Drosophila Neurons**: 정량적 평가는 불가능했으나, 매우 얇고 복잡하게 얽힌 신경세포 구조를 성공적으로 분할해내는 정성적 결과를 보여주어 범용성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **End-to-End 학습의 중요성**: `ppp+ae`보다 `ppp+dec`가 우수한 성능을 보인 것은, 인코더(U-Net)와 디코더가 인스턴스 분할 작업에 최적화되도록 함께 학습되는 것이 유리함을 시사한다.
- **합의 메커니즘의 효과**: 단순히 CNN의 예측값을 그대로 사용하는 대신, 주변 패치들과의 합의(Consensus)를 통해 패치를 선택하는 과정이 노이즈를 제거하고 객체의 연속성을 확보하는 데 핵심적인 역할을 한다.
- **U-Net의 우위**: 슬라이딩 윈도우 방식의 `ed-ppp`보다 전역적 문맥을 활용하는 U-Net 기반의 `ppp+dec`가 훨씬 뛰어난 성능을 보였다.

### 한계 및 미해결 질문

- **3D 계산 병목**: 3D 데이터에서 패치 크기가 커질수록 인스턴스 조립 단계의 계산 시간이 급격히 증가한다. 현재는 성능을 위해 패치 크기를 작게 유지하고 있으나, 이는 최적의 성능을 내지 못하는 원인이 된다.
- **겹침 영역의 제한**: 본 방법론은 패치 크기보다 작은 직경의 겹침 영역만 처리할 수 있으며, 패치 그래프 생성 시 설정한 이웃 범위 내의 폐색(Occlusion)만 극복 가능하다는 한계가 있다.

## 📌 TL;DR

본 논문은 **학습된 국소 형상 패치를 예측하고 이를 합의 기반으로 조립하여 인스턴스를 분할하는 PatchPerPix**를 제안한다. 이 방법은 특히 생물 의학 이미지에서 나타나는 복잡한 형태, 밀집된 클러스터, 객체 간 교차 문제를 효과적으로 해결하며, 비반복적인 One-pass 파이프라인을 통해 모든 인스턴스를 동시에 분할한다. 다양한 벤치마크에서 SOTA를 달성함으로써 범용적인 인스턴스 분할 도구로서의 가능성을 보여주었으며, 향후 3D 데이터 처리 효율성 개선이 주요 연구 과제로 남아 있다.
