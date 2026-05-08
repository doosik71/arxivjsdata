# Zero-Shot Anomaly Detection with Pre-trained Segmentation Models

Matthew Baugh, James Batten, Johanna P. Müller, and Bernhard Kainz (2023)

## 🧩 Problem to Solve

본 논문은 산업용 이상 탐지(Industrial Anomaly Detection) 분야에서 기존 모델들이 직면한 데이터 의존성 문제를 해결하고자 한다. 현재의 비지도 학습(Unsupervised Learning) 기반 이상 탐지 방법들은 MVTec과 같은 데이터셋에서 매우 높은 성능(AUROC 98.0 이상)을 보이지만, 이를 위해서는 대량의 정상 샘플(Normal samples)이 필요하다는 한계가 있다.

실제 산업 현장에서는 모든 제품의 정상 데이터를 충분히 확보하는 것이 어려우며, 따라서 정상 데이터가 거의 없거나 전혀 없는 상태에서도 이상을 탐지할 수 있는 데이터 효율적인 방법론이 필요하다. 특히 기존의 Zero-Shot 방법론인 WinCLIP이 제안되었으나, 이상 부위를 정밀하게 찾아내는 Localization(국소화) 능력이 부족하다는 점이 문제로 지적되었다. 본 연구의 목표는 사전 학습된 세그멘테이션 모델들을 통합하여 Zero-Shot 환경에서 이상 탐지의 국소화 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 학습된 Foundation Model(기반 모델)들을 조합하여, 별도의 추가 학습 없이도 객체의 전경(Foreground)을 정밀하게 추출하고 이를 기반으로 이상 부위를 세밀하게 식별하는 파이프라인을 구축하는 것이다.

주요 기여 사항은 다음과 같다.

1. **전경 추출(Foreground Extraction)의 정밀화**: Dichotomous Image Segmentation(DIS)과 Segment Anything Model(SAM)을 결합하여 객체 본연의 영역만을 정확히 분리함으로써, 배경 노이즈나 그림자로 인한 오탐지를 줄였다.
2. **타일 기반 분석(Image Tiling) 전략**: 이미지를 고정된 크기로 나누는 대신, 전경 객체의 연결 성분(Connected Component)을 중심으로 타일을 생성하여 작은 이상 징후를 더 잘 포착할 수 있게 하였다.
3. **계층적 예측 구조**: Tile-level Predictor와 Pixel-level Predictor를 분리하여 운영하고, 타일 수준의 신뢰도로 픽셀 수준의 예측값을 보정하는 예측 집계(Prediction Aggregation) 방식을 도입하였다.

## 📎 Related Works

본 연구는 WinCLIP 프레임워크를 기반으로 한다. WinCLIP은 언어 가이드(Language Guidance)를 활용하여 정상 이미지 없이도 CLIP 모델의 임베딩을 통해 이상 여부를 판별하고 국소화하는 Zero-Shot 방법론이다.

기존 접근 방식과의 차별점은 다음과 같다.

- **Localization 능력 강화**: WinCLIP은 이미지 전체를 설명하는 프롬프트를 사용하여 세그멘테이션을 수행하므로 결과가 너무 광범위하게 나타나는 경향이 있다. 본 논문은 이를 해결하기 위해 이상 부위만을 묘사하는 'Localising prompts'와 CLIPSeg 모델을 도입하였다.
- **객체 중심 분석**: 이미지 전체를 처리하는 대신, SAM과 DIS를 통해 추출된 전경 인스턴스별로 분석을 수행함으로써 다중 객체가 존재하는 상황에서도 개별 객체의 미세한 결함을 더 효과적으로 찾아낼 수 있도록 설계하였다.

## 🛠️ Methodology

전체 파이프라인은 전경 추출, 이미지 타일링, 프롬프트 생성, 예측(타일 및 픽셀 수준), 예측 집계의 5단계로 구성된다.

### 1. Foreground Extraction

객체의 전경을 식별하기 위해 DIS(Dichotomous Image Segmentation)와 SAM(Segment Anything Model)을 함께 사용한다. DIS를 통해 생성된 마스크를 필터로 사용하여 SAM이 생성한 여러 어노테이션 중 전경에 해당하는 것만 선택한다. 구체적으로 SAM 마스크의 80% 이상이 DIS 마스크에 의해 커버되는 경우에만 전경으로 인정하며, 이를 통해 SAM이 객체의 그림자까지 포함하여 영역을 과하게 잡는 문제를 방지한다.

### 2. Image Tiling

전경 마스크의 각 연결 성분을 중심으로 정사각형 타일을 생성한다.

- **기본 설정**: 최소 해상도를 $352 \times 352$로 설정하여 모델의 전처리 과정에서 발생하는 이미지 왜곡을 방지한다.
- **특수 처리**: 종횡비가 1.5보다 크고 구성 요소가 많은 객체의 경우, 긴 축을 따라 짧은 축 길이의 절반 간격으로 타일을 슬라이딩하며 생성하여 객체의 모든 부분이 중심에 포함되도록 한다.

### 3. Prompt Generator

- **Sample-level prompts**: WinCLIP의 조합적 프롬프트 앙상블을 유지하되, 견고성을 높이기 위해 정상/비정상 상태를 나타내는 단어 리스트(예: "pristine", "shattered" 등)를 확장하였다.
- **Localising prompts**: "a tear", "a crack", "a scratch"와 같이 이상 부위의 일반적인 명사들을 사용하여, 이미지 전체가 아닌 특정 결함 영역만을 세그멘테이션 하도록 유도하였다.

### 4. Predictors

#### Tile-level Predictor

각 타일의 이상 점수를 계산하기 위해 CLIP 임베딩을 사용한다. 기존 WinCLIP이 프롬프트 임베딩들의 평균을 먼저 구한 뒤 이미지와 비교했다면, 본 방법은 이미지 임베딩과 각 프롬프트 임베딩 간의 코사인 유사도를 먼저 구한 뒤 그 값들을 평균낸다. 이는 비정상 상태를 묘사하는 프롬프트들이 서로 매우 이질적이기 때문에, 임베딩 공간에서 평균을 내는 것보다 유사도 공간에서 평균을 내는 것이 더 의미 있다는 직관에 근거한다.

#### Pixel-level Predictor

각 타일에 대해 CLIPSeg 모델을 적용하고 Localising prompts를 입력하여 세그멘테이션 맵을 생성한다. 여러 프롬프트로부터 얻은 결과물들에 대해 조화 평균(Harmonic Average)을 적용하여, 여러 프롬프트에서 공통적으로 높은 활성화 값을 보이는 영역에 집중하도록 한다.

### 5. Prediction Aggregation

타일 수준의 예측이 픽셀 수준보다 더 정확하고 견고하므로, 픽셀 예측값에 해당 타일의 이상 점수를 가중치로 곱하여 스케일링한다.

- **Sample-level prediction**: 전경 구성 요소별로 타일 점수를 평균낸 후, 그중 상위 25%의 점수만을 평균하여 최종 샘플 점수를 산출한다. 이는 다중 객체 이미지에서 정상 영역의 노이즈를 제거하기 위한 전략이다.

## 📊 Results

### 실험 설정

- **데이터셋**: VisA 데이터셋
- **측정 지표**: $F1\text{-max}$ (임계값을 최적화했을 때의 F1-score). 이는 이상 부위가 이미지 전체 크기에 비해 매우 작아 발생하는 클래스 불균형 문제에 강건하기 때문에 선택되었다.
- **비교 대상**: WinCLIP (Baseline), APRIL-GAN (VAND 챌린지 우승 모델)

### 주요 결과

- **Sample-wise F1-max**: 본 제안 방법은 **81.5**를 기록하며 WinCLIP(79.0)과 APRIL-GAN(78.7)을 모두 제치고 새로운 SOTA 성능을 달성하였다.
- **Pixel-wise F1-max**: 본 제안 방법은 **24.2**를 기록하여 WinCLIP(14.8)보다는 크게 향상되었으나, APRIL-GAN(32.3)에는 미치지 못하였다.

| 방법론 | Sample-wise $F1\text{-max}$ | Pixel-wise $F1\text{-max}$ |
| :--- | :---: | :---: |
| WinCLIP | 79.0 | 14.8 |
| APRIL-GAN | 78.7 | 32.3 |
| **Proposed (VVV)** | **81.5** | **24.2** |

## 🧠 Insights & Discussion

본 논문은 사전 학습된 세그멘테이션 모델(SAM, CLIPSeg)을 통합함으로써 WinCLIP의 가장 큰 약점이었던 국소화 능력을 유의미하게 개선하였다. 특히 전경 추출 단계를 통해 분석 범위를 객체로 한정 지은 점이 성능 향상에 기여하였다고 평가할 수 있다.

하지만 다음과 같은 한계점이 존재한다.

1. **Domain Shift 문제**: 사용된 기반 모델들이 일반적인 이미지 데이터로 학습되었기 때문에, 매우 특수한 산업용 데이터셋으로 넘어왔을 때 발생하는 도메인 간 차이(Domain Shift)로 인해 성능 저하가 발생한다.
2. **미세 결함 탐지**: 이상 부위가 극도로 작거나 미묘한 경우, 현재의 Zero-Shot 세그멘테이션 모델들로는 이를 완전히 포착하는 데 한계가 있다.
3. **성능 격차**: 샘플 수준의 분류 성능은 향상되었으나, 픽셀 수준의 정밀도는 여전히 비지도 학습 모델이나 일부 특화된 Zero-Shot 모델(APRIL-GAN)에 비해 낮다.

결론적으로, Foundation Model의 표현력을 활용하는 방향은 유효하지만, 산업 현장의 특수성을 반영한 더 정교한 도메인 적응 또는 더 고해상도의 표상 학습이 필요함을 시사한다.

## 📌 TL;DR

본 논문은 WinCLIP의 Zero-Shot 이상 탐지 프레임워크에 SAM과 CLIPSeg 같은 최신 세그멘테이션 모델을 결합하여, 전경 추출 및 타일 기반 분석을 통해 이상 부위 국소화(Localization) 성능을 높인 연구이다. VisA 데이터셋의 샘플 수준 탐지에서 SOTA 성능(F1-max 81.5)을 달성하였으며, 이는 추가 학습 없이 기반 모델들의 조합만으로도 산업용 이상 탐지의 효율성을 높일 수 있음을 보여준다. 향후 연구에서는 산업 도메인 특화 모델의 필요성이 강조된다.
