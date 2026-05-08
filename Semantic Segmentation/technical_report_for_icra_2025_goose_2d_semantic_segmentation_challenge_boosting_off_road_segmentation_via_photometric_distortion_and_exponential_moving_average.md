# Technical Report for ICRA 2025 GOOSE 2D Semantic Segmentation Challenge: Boosting Off-Road Segmentation via Photometric Distortion and Exponential Moving Average

Wonjune Kim, Lae-Kyoung Lee and Su-Yong An (2025)

## 🧩 Problem to Solve

본 논문은 비정형 오프로드(unstructured off-road) 환경에서 자율 주행을 위한 2D Semantic Segmentation 문제를 해결하고자 한다. 오프로드 환경은 도시 환경과 달리 도로 경계석이나 차선과 같은 구조적 단서가 부족하며, 진흙, 눈, 밀집된 덤불 등 외관의 다양성이 매우 크다. 특히 급격하게 변하는 날씨와 조명 조건은 인식 시스템의 정확도를 떨어뜨리는 주요 요인이 된다.

연구진이 정의한 이 문제의 핵심 난제는 다음과 같다.
첫째, 심각한 클래스 불균형(Class Imbalance) 문제이다. 전체 픽셀의 약 90%가 vegetation, terrain, sky 세 가지 클래스에 집중되어 있으며, 정작 안전에 치명적인 obstacle나 human과 같은 클래스는 데이터 양이 매우 적어 학습이 어렵다.
둘째, 모호하고 낮은 대비의 경계(Ambiguous, low-contrast boundaries) 문제이다. 자연물(예: 풀과 흙, 물과 진흙)은 서로 경계가 점진적으로 변하는 특성이 있어, 일반적인 Edge 기반 세그멘테이션 기법이나 표준 Cross-Entropy 최적화 방식으로는 명확한 구분이 어렵다.

따라서 본 논문의 목표는 이러한 오프로드 환경의 특수성을 극복하여, 특히 소수 클래스에 대한 인식 성능을 높이고 조명 변화에 강건한 세그멘테이션 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 새로운 네트워크 아키텍처를 설계하는 대신, 검증된 고성능 컴포넌트들을 오프로드 환경의 특성에 맞게 정밀하게 조합하고 최적화하는 '학습 레시피'를 제안하는 것이다.

가장 중점적인 기여는 강한 Photometric Distortion 증강 기법과 가중치의 Exponential Moving Average(EMA)를 결합하여 일반화 성능을 극대화한 점이다. Photometric Distortion을 통해 야외 지형의 극심한 조명 변화를 모사함으로써 모델이 단순한 색상 정보가 아닌 형태와 질감에 집중하게 만들고, EMA를 통해 레이블 노이즈를 완화하고 학습의 안정성을 확보하여 최종적으로 세그멘테이션 경계를 더욱 정교하게 다듬었다.

## 📎 Related Works

본 연구는 오프로드 환경의 벤치마킹을 위해 제공된 GOOSE 및 GOOSE-EX 데이터셋을 기반으로 한다. 이 데이터셋들은 계절별로 다양한 RGB 프레임과 세밀한 라벨 맵을 제공하며, 본 챌린지에서는 이를 9개의 운영 카테고리로 통합하여 사용한다.

기존의 접근 방식들이 주로 새로운 모델 구조를 제안하는 데 집중했다면, 본 논문은 FlashInternImage-B와 UPerNet과 같은 최신 고용량(high-capacity) 백본과 디코더를 채택하고, 이를 보조하는 학습 전략(Augmentation, EMA)의 시너지를 통해 성능을 끌어올리는 실용적인 접근 방식을 취한다. 특히 DCNv4와 같은 최신 연산자를 도입하여 효율성과 정확도를 동시에 확보하고자 하였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문은 $\text{FlashInternImage-B}$ 백본과 $\text{UPerNet}$ 디코더를 결합한 파이프라인을 사용한다.

1. **Backbone ($\text{FlashInternImage-B}$):** $\text{InternImage-B}$의 모든 Deformable Convolution 레이어를 $\text{DCNv3}$에서 더 빠른 $\text{DCNv4}$로 업그레이드한 모델이다. 이를 통해 정확도는 유지하면서 학습 속도를 약 1.8배 향상시켰다.
2. **Decoder ($\text{UPerNet}$):** 입력 해상도 대비 $\frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}$ 크기의 피처 맵을 집계한다. $\text{FPN}$ 브랜치를 통해 다중 스케일 정보를 병합하고, $\text{PSP}$ 브랜치를 통해 글로벌 컨텍스트를 캡처하여 최종적으로 9개의 클래스에 대한 로짓(logits)을 생성한다.

### 학습 절차 및 손실 함수

- **최적화:** $\text{AdamW}$ 옵티마이저(초기 학습률 $6 \times 10^{-5}$)를 사용하며, $\text{poly}$ 학습률 스케줄에 따라 총 $96\text{k}$ 반복 학습을 수행한다.
- **입력 데이터:** 이미지를 $[0.5, 2.0]$ 범위에서 랜덤하게 스케일링한 후 $2048 \times 2048$ 크기로 크롭하거나 패딩하여 사용한다.
- **손실 함수:** 픽셀 단위의 $\text{soft-max cross-entropy}$를 최적화 목표로 사용한다.

### 핵심 학습 전략

**1. Photometric Distortion**
야외 환경의 극심한 조명 변화(어두운 숲 $\leftrightarrow$ 밝은 설원)에 대응하기 위해, 학습 시 다음의 네 가지 요소를 각각 0.5의 확률로 독립적으로 변형시킨다.

- Brightness (밝기)
- Contrast (대비)
- Saturation (채도)
- Hue (색조)
이러한 무작위 색상 변형은 네트워크가 원시 색상 정보보다는 형태(Shape)와 질감(Texture)에 의존하도록 강제한다.

**2. Exponential Moving Average (EMA)**
학습의 안정성을 높이고 레이블 노이즈를 완화하기 위해 네트워크 파라미터의 EMA를 유지한다. 매 반복마다 다음과 같은 수식으로 가중치를 업데이트한다.

$$\theta_{EMA}^{(t)} = \alpha \theta_{EMA}^{(t-1)} + (1-\alpha)\theta_{CURRENT}^{(t)}, \quad \alpha = 0.999$$

검증 및 최종 평가 단계에서는 $\theta_{CURRENT}$가 아닌 $\theta_{EMA}$ 스냅샷을 사용하여 추론을 수행한다.

## 📊 Results

### 실험 설정

- **데이터셋:** $\text{GOOSE}$ 학습 셋($\approx 8\text{k}$ 이미지)과 $\text{GOOSE-EX}$ 학습 셋($\approx 4\text{k}$ 이미지)을 통합하여 학습하고, 공식 검증 셋($\approx 1.4\text{k}$ 이미지)으로 평가하였다.
- **지표:** 9개 클래스에 대한 평균 $\text{mIoU (mean Intersection-over-Union)}$를 측정하였다.
- **하드웨어:** 4대의 NVIDIA RTX 3090 GPU를 사용하였다.

### 정량적 결과

실험 결과, 기본 모델에서 제안 기법을 순차적으로 적용했을 때 성능이 지속적으로 향상되었다.

| 모델 구성 | mIoU ($\uparrow$) | 특이사항 |
| :--- | :---: | :--- |
| $\text{FlashInternImage-B}$ (Baseline) | 87.2% | 기본 성능 |
| + Photometric Distortion | 87.76% | $+0.48$ mIoU 상승, $\text{sky}$ 및 $\text{other}$ 클래스 개선 |
| + Photometric Distortion + EMA | **88.88%** | $+1.12$ mIoU 추가 상승 (총 $+1.60$), $\text{obstacle}$ 및 $\text{human}$ 클래스 크게 개선 |

최종적으로 ICRA 2025 GOOSE 2D 챌린지 공식 테스트 서버에서 **84.5 mIoU**를 기록하며 공개 리더보드 **2위**를 달성하였다.

### 정성적 결과

- **Photometric Distortion의 효과:** 기본 모델이 도로 우측의 바위 더미를 $\text{natural ground}$로 오분류한 반면, 해당 기법을 적용한 모델은 이를 $\text{obstacle}$로 정확히 분류하였다. 또한 $\text{artificial ground}$와 $\text{natural ground}$를 더 명확하게 구분하는 능력을 보였다.
- **EMA의 효과:** 균일한 대형 영역에서 발생하는 스펙클 아티팩트(speckle artifacts)를 억제하고, 클래스 간의 경계를 더욱 날카롭고 명확하게 생성하는 효과가 확인되었다.

## 🧠 Insights & Discussion

본 논문은 모델의 구조적 변경보다 데이터 증강과 최적화 전략이 오프로드 환경과 같은 특수한 도메인에서 얼마나 중요한지를 보여준다. 특히 다음과 같은 통찰을 얻을 수 있다.

첫째, $\text{Photometric Distortion}$은 모델이 색상에 과적합(overfitting)되는 것을 방지하여, 조명 변화가 심한 야외 환경에서 강건함을 제공한다. 이는 정량적으로는 소폭의 상승이지만, 정성적으로는 $\text{obstacle}$과 같은 중요한 클래스의 오분류를 줄이는 실질적인 효과를 낸다.

둘째, $\text{EMA}$는 단순히 수렴 속도를 돕는 것을 넘어, 데이터셋에 존재하는 레이블 노이즈를 완화하고 결과물의 시각적 품질(경계면의 선명도)을 높이는 역할을 한다. 특히 데이터 양이 적은 소수 클래스($\text{obstacle, human}$)의 성능 향상에 기여했다는 점이 주목할 만하다.

다만, 본 논문은 기존의 고성능 모델(FlashInternImage-B)과 기법들을 조합한 형태이므로, 완전히 새로운 알고리즘적 기여보다는 '최적의 조합'을 찾아낸 엔지니어링적 성과에 가깝다. 또한, 사용된 모델의 연산량이 매우 높을 것으로 예상되나, 실제 실시간 시스템에 적용했을 때의 추론 속도(FPS)에 대한 분석은 명시되지 않았다.

## 📌 TL;DR

본 연구는 오프로드 환경의 세그멘테이션 난제인 조명 변화와 클래스 불균형을 해결하기 위해 $\text{FlashInternImage-B} + \text{UPerNet}$ 구조에 $\text{Photometric Distortion}$과 $\text{EMA}$를 적용한 파이프라인을 제안하였다. 이를 통해 검증 셋 기준 $88.88\%\text{ mIoU}$를 달성하였으며, 특히 소수 클래스 인식률과 경계면 선명도를 크게 개선하여 ICRA 2025 챌린지에서 2위를 기록하였다. 이 연구는 비정형 환경의 인지 시스템 구축 시 강력한 증강 기법과 가중치 평균화 전략이 필수적임을 시사한다.
