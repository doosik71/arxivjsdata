# Segment Any Medical Model Extended

Yihao Liu, Jiaming Zhang, Andrés Diaz-Pinto, Haowei Li, Alejandro Martin-Gomez, Amir Kheradmand, and Mehran Armand (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 방사선학, 병리학, 내시경 및 영상 유도 치료 등 다양한 임상 분야에서 필수적인 작업이다. 그러나 의료 영상 분할은 다음과 같은 몇 가지 핵심적인 문제점을 가지고 있다.

첫째, 의료 영상 데이터는 기본적으로 3D 볼륨(volumetric) 형식이지만, 최근 주목받는 Foundation Model인 Segment Anything Model (SAM)은 2D 이미지 분할을 위해 설계되었다. 따라서 3D 데이터를 분할하기 위해서는 슬라이스별(slice-by-slice) 접근 방식이 강제되며, 이는 매우 노동 집약적이고 지루한 과정이다.

둘째, Vanilla SAM은 일반적인 이미지에 대해서는 강력한 Zero-shot 성능을 보이지만, 의료 영상 도메인에서는 SOTA(State-of-the-art) 비-파운데이션 모델들에 비해 성능이 제한적이라는 사실이 밝혀졌다.

셋째, 기존의 SAM 기반 의료 영상 도구들은 추론 속도가 느려 실시간 상호작용이 어렵고, 다양한 SAM 변형 모델(variants)들을 통합하여 비교·검증할 수 있는 통일된 플랫폼이 부족한 실정이다.

본 논문의 목표는 이러한 한계를 극복하기 위해 SAM의 변형 모델들을 통합하고, 빠른 통신 프로토콜과 새로운 상호작용 모드를 지원하며, 모델의 세부 구성 요소를 미세 조정(fine-tuning)할 수 있는 통합 플랫폼인 SAMME (Segment Any Medical Model Extended)를 제공하는 것이다.

## ✨ Key Contributions

본 연구의 중심 아이디어는 SAM의 구조적 특성(Image Encoder $\rightarrow$ Mask Decoder)을 활용하여 3D 의료 영상 환경에서의 상호작용 효율성을 극대화하는 것이다. 주요 기여 사항은 다음과 같다.

- **실시간 추론을 위한 Save-and-Retrieve 패러다임**: 모든 슬라이스의 Embedding(특징 벡터)을 사전에 계산하여 메모리에 저장함으로써, 프롬프트 입력 시 Mask Decoder만 실행하여 즉각적인 결과(0.1초 미만)를 도출한다.
- **Prompt Propagation (프롬프트 전파)**: 인접한 슬라이스 간의 유사성을 이용하여, 한 슬라이스에서 설정한 프롬프트를 다음 슬라이스로 그대로 전달함으로써 반복적인 프롬프트 입력을 제거한다.
- **3D Bounding Box 지원**: 사용자가 3개 축의 뷰에서 3D 바운딩 박스를 설정하면, 해당 범위 내의 모든 슬라이스에 대해 자동으로 분할을 수행하는 완전 자동화 모드를 제공한다.
- **통합 플랫폼 구축**: MedSAM, MobileSAM 등 다양한 SAM 변형 모델을 쉽게 통합할 수 있는 구조를 설계하고, 3D Slicer와의 연동을 통해 임상 활용 가능성을 높였다.

## 📎 Related Works

기존 연구들은 SAM을 의료 영상에 적용하기 위해 주로 다음과 같은 방향으로 접근하였다.

- **Fine-tuning**: MedSAM과 같이 의료 영상 데이터셋으로 SAM을 미세 조정하여 도메인 적응력을 높이려는 시도가 있었다.
- **Efficiency**: MobileSAM과 같이 이미지 인코더를 경량화하여 추론 속도를 개선하려는 연구가 진행되었다.
- **Evaluation**: SAM의 Zero-shot 성능을 의료 영상에서 평가하고 그 한계를 분석한 연구들이 다수 발표되었다.

기존 접근 방식의 한계는 이러한 개별 모델의 개선에만 집중했을 뿐, 실제 의료 영상 분석가들이 사용할 수 있는 3D 인터페이스 및 실시간 상호작용 워크플로우에 대한 통합적인 솔루션을 제시하지 못했다는 점이다. SAMME는 이전 작업인 SAMM을 확장하여, 단순한 모델 적용을 넘어 실시간성, 3D 확장성, 모델 교체 가능성을 모두 갖춘 플랫폼을 지향함으로써 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
SAMME는 크게 세 가지 주요 구성 요소로 이루어져 있다 (Figure 3 참조).
1. **3D Slicer**: 사용자 인터페이스(GUI), 데이터 저장, 시각화 및 렌더링을 담당하는 오픈 소스 소프트웨어이다.
2. **SAMME Bridge**: 3D Slicer의 이미지 좌표 데이터를 SAM 모델이 이해할 수 있는 좌표로 해석하고 변환하는 중간 계층이다.
3. **SAMME Server**: 모델 계산 및 마스크 예측을 수행하는 Task Queue를 운영하며, 실제 SAM 모델들이 구동되는 서버이다.

### 핵심 메커니즘

**1. Embedding Precomputation (임베딩 사전 계산)**
SAM의 구조는 매우 무거운 Image Encoder와 상대적으로 가벼운 Mask Decoder로 나뉜다. SAMME는 추론 요청 시마다 전체 모델을 돌리는 대신, 이미지의 모든 슬라이스에 대해 Image Encoder를 먼저 실행하여 Embedding을 계산하고 이를 메모리에 저장한다. 이후 사용자가 프롬프트를 입력하면 저장된 Embedding과 프롬프트 정보만을 사용하여 Mask Decoder를 실행하므로 실시간 응답이 가능하다.

**2. Prompt Propagation (프롬프트 전파)**
의료 영상의 인접 슬라이스는 해부학적으로 매우 유사하다. 이를 이용하여 사용자가 마우스 휠로 슬라이스를 이동할 때, 이전 슬라이스에서 사용한 포인트나 바운딩 박스 프롬프트를 다음 슬라이스에 자동으로 적용한다.

**3. 3D Bounding Box**
사용자가 세 가지 해부학적 뷰(Axial, Sagittal, Coronal)에서 3D 영역을 지정하면, 시스템은 이를 2D 바운딩 박스의 연속적인 전파로 해석하여 해당 볼륨 내 모든 슬라이스의 마스크를 자동으로 생성한다.

**4. 모델 통합 조건**
SAMME는 다음과 같은 조건을 만족하는 모든 SAM 변형 모델을 통합할 수 있다.
- 학습된 가중치(Weights)가 제공될 것.
- SAM의 `SamPredictor` 클래스 인터페이스를 통해 추론이 가능할 것.
- $\text{Prompt Encoder} \rightarrow \text{Image Encoder} \rightarrow \text{Mask Decoder}$ 구조를 따를 것.

## 📊 Results

### 실험 설정 및 지표
- **하드웨어**: Ubuntu 20.04, AMD Ryzen 9 3900X, Nvidia GeForce RTX 3090.
- **테스트 데이터**: 3D Slicer 샘플 데이터 (256 $\times$ 256 $\times$ 130 볼륨).
- **평가 모델**: MobileSAM, MedSAM, Vanilla SAM (vit-b, vit-l, vit-h).
- **측정 지표**: 마스크 오버레이 시간(Mask overlay time), 추론 시간(Inference time), 임베딩 계산 시간(Embedding calculation time).

### 정량적 결과 (Table 1)
- **전체 사이클 시간**: 프롬프트 전송부터 마스크 시각화까지의 전체 시간은 약 $0.06$초로, 이전 버전인 SAMM($0.6$초) 대비 10배 향상되었다.
- **추론 속도**: 실제 모델의 추론 시간(Inference time)은 모델 종류에 관계없이 약 $0.008$초로 매우 빠르게 나타났다.
- **임베딩 계산 시간**: 모델의 크기에 따라 차이가 있으며, MobileSAM($7.231$s)이 가장 빠르고 Vanilla vit-h($146.783$s)가 가장 느렸다.

### 정성적 결과 및 분석
- **모델별 성능**: Figure 5에서 확인되듯, 동일한 바운딩 박스 프롬프트에 대해서도 모델마다 분할 결과에 차이가 발생한다.
- **전처리의 영향**: 특히 Window/Level 값(밝기 및 대비 설정)에 따라 분할 결과가 민감하게 변하는 것이 확인되었다(Figure 4b). 이는 SAM 변형 모델을 학습시키거나 사용할 때 의료 영상 특유의 윈도잉 처리가 중요함을 시사한다.

## 🧠 Insights & Discussion

**강점**
SAMME는 단순한 알고리즘 제안이 아니라, 실제 의료 영상 분석가들이 겪는 '반복 작업의 고통'을 시스템적으로 해결하려 했다는 점에서 실용적 가치가 높다. 특히 Save-and-Retrieve 패러다임을 통해 Foundation Model의 무거운 연산 비용 문제를 해결하고 실시간 상호작용을 가능케 한 점이 돋보인다.

**한계 및 논의사항**
1. **프롬프트의 주관성**: 논문에서도 언급되었듯, 프롬프트를 어떻게 입력하느냐에 따라 결과가 달라지며, 이에 대한 일관된 가이드라인이 부족하다.
2. **전처리 일관성**: MedSAM 등 일부 모델은 내부적으로 $1024 \times 1024$ 리사이징을 수행하지만, SAMME는 원본 크기를 유지한다. 이러한 전처리 과정의 차이가 실제 성능 편차를 야기할 수 있으며, 이에 대한 정밀한 비교 분석이 필요하다.
3. **3D 맥락 부족**: 본 시스템은 2D 슬라이스의 연속적인 처리를 효율화한 것이지, 모델 자체가 3D 공간 정보(volumetric context)를 직접 학습하여 추론하는 것은 아니다.

## 📌 TL;DR

본 논문은 2D 기반의 SAM을 3D 의료 영상 분할에 효율적으로 적용하기 위한 통합 플랫폼 **SAMME**를 제안한다. 임베딩 사전 계산을 통해 실시간 추론($0.06$s)을 구현하였으며, 프롬프트 전파 및 3D 바운딩 박스 기능을 통해 수동 작업량을 획기적으로 줄였다. 이 플랫폼은 다양한 SAM 변형 모델을 통합하여 비교할 수 있게 함으로써, 향후 의료 영상 기반의 데이터 증강, 로봇 내비게이션, 영상 유도 치료 등의 응용 분야에 핵심적인 도구로 활용될 가능성이 크다.