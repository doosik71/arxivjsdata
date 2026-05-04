# Insight Any Instance: Promptable Instance Segmentation for Remote Sensing Images

Xuexue Li, Wenhui Diao, Xinming Li and Xian Sun(2024)

## 🧩 Problem to Solve

원격 탐사 이미지(Remote Sensing Images, RSIs)에서의 인스턴스 분할(Instance Segmentation)은 토지 계획 및 지능형 교통 시스템 등 다양한 응용 분야에서 매우 중요하다. 그러나 RSIs는 일반적인 자연 이미지(Natural Scenes)와 비교하여 다음과 같은 고유한 문제점을 가지고 있다.

1. **심각한 전경-배경 불균형**: RSIs에서는 전경 픽셀의 비율이 매우 낮으며, 많은 이미지에서 전경 픽셀이 10% 미만을 차지한다.
2. **제한된 인스턴스 크기**: 대상 객체의 크기가 매우 작아 정밀한 분할이 어렵다.
3. **딥러닝 구조의 한계**: 현재 대부분의 인스턴스 분할 모델은 딥러닝 기반의 특징 추출 방식을 사용하며, 이 과정에서 다수의 다운샘플링(Downsampling) 연산이 포함된다. 이러한 연산은 공간적 세부 정보를 손실시켜, 특히 작은 객체가 많은 RSIs의 인스턴스 분할 성능을 저하시키는 결정적인 원인이 된다.

본 논문의 목표는 다운샘플링으로 인한 정보 손실을 보완하고, 전경-배경 불균형 및 작은 객체 문제를 해결하기 위해 새로운 **프롬프트 패러다임(Prompt Paradigm)**을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 다운샘플링이 적용되지 않은 원본 이미지에서 인스턴스 특화 프롬프트 정보를 직접 추출하여, 기존 모델의 디코더에 제공함으로써 공간 정보의 손실을 보완하는 것이다.

- **Local Prompt Module (LPM)**: 원본 이미지의 국소 영역에서 텍스트 및 구조 정보를 마이닝하여 인스턴스의 표현력을 높인다.
- **Global-to-Local Prompt Module (GPM)**: 전역 토큰에서 국소 토큰으로의 컨텍스트 정보를 모델링하여, 작은 객체가 부족한 자체 텍스트 정보를 전역 문맥으로 보완한다.
- **Proposal's Area Loss (PAreaLoss)**: 제안된 영역(Proposal)의 위치뿐만 아니라 스케일(Scale) 차원을 분리하여 최적화함으로써, 더 정확한 Proposal을 생성하고 이를 통해 LPM과 GPM의 성능을 극대화한다.
- **Promptable Instance Segmentation 확장**: 제안된 구조를 통해 박스 프롬프트(Box Prompt)를 입력받아 특정 객체를 분할하는 인터랙티브한 프롬프트 기반 분할 기능을 제공한다.

## 📎 Related Works

### 기존 연구 및 한계

- **일반 인스턴스 분할 모델**: Mask R-CNN, Cascade Mask R-CNN 등은 자연 이미지에서는 우수하지만, RSIs의 복잡한 배경과 스케일 변화에 취약하다.
- **원격 탐사 특화 모델**: SCBN, RSIISN 등은 스케일 변화와 복잡한 배경 문제를 해결하려 했으나, 여전히 딥러닝 특징 추출의 다운샘플링 패러다임에 의존하고 있어 작은 객체에 대한 정보 손실 문제를 근본적으로 해결하지 못했다.
- **고해상도 유지 모델**: HRNet과 같이 다운샘플링을 제거한 모델이 존재하지만, 이는 연산량이 너무 방대하여 실제 응용에 제약이 있다.
- **프롬프트 학습(Prompt Learning)**: SAM(Segment Anything Model)은 프롬프트를 통한 분할의 가능성을 보여주었으나, 파라미터 수가 너무 많아 실제 원격 탐사 장비에 배포하기 어렵다.

### 차별점

본 논문은 모든 픽셀을 고해상도로 유지하는 대신, **인스턴스가 위치한 특정 영역에 대해서만 프롬프트 형태로 고해상도 정보를 주입**함으로써 연산 효율성을 유지하면서도 다운샘플링의 폐해를 극복했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 문제의 재정의 (Rethink)

저자들은 일반적인 특징 추출 과정에서의 정보 손실을 다음과 같이 모델링한다.
$$F_i = e^{\beta \cdot i} \cdot e^{-\alpha \cdot s_i^2} \cdot I$$
여기서 $e^{\beta \cdot i}$는 채널 차원의 정보 이득을, $e^{-\alpha \cdot s_i^2}$는 공간 차원의 정보 감쇠(Decay)를 나타낸다. 기존 모델 $R = D(\sum F_i)$는 이 감쇠 효과로 인해 성능이 제한된다.

본 논문은 다운샘플링이 없는 프롬프트 인코더 $P = e^\gamma \cdot I$를 도입하여, 최종 결과 $R$을 다음과 같이 도출한다.
$$R = D(\sum F_i, P) = D(\sum e^{\beta \cdot i} \cdot I \cdot e^{-\alpha \cdot s_i^2}, e^\gamma \cdot I)$$
즉, 기존의 특징 맵과 다운샘플링되지 않은 프롬프트 정보를 함께 디코더에 입력하는 구조이다.

### 2. Local Prompt Module (LPM)

LPM은 인스턴스가 위치한 국소 영역의 풍부한 텍스트 정보를 추출한다.

- **절차**: 원본 이미지에서 Proposal 기반으로 국소 영역을 크롭 $\rightarrow$ 겹치는 패치 분할(Overlapping Patch Partition) $\rightarrow$ 고속 푸리에 변환(FFT)을 통한 주파수 영역 상호작용 $\rightarrow$ 학습 가능한 임베딩 추가 $\rightarrow$ 역고속 푸리에 변환(IFFT) $\rightarrow$ MLP 레이어를 통한 최종 Local Prompt 생성.
- **효과**: 배경 영역을 제외하고 인스턴스 영역만 집중적으로 처리하므로 배경 노이즈의 간섭을 피하고 강건성을 높인다.

### 3. Global-to-Local Prompt Module (GPM)

작은 객체는 자체 텍스트 정보가 부족하므로 전역 문맥(Global Context)을 활용한다.

- **구조**: 원본 이미지를 전역 토큰($G$)과 인스턴스 위치의 국소 토큰($L$)으로 나눈다.
- **연산**: Transformer의 Multi-head Self-Attention을 통해 전역-국소 상호작용을 수행하며, 다음과 같은 방정식으로 국소 토큰 $\tilde{L}$을 업데이트한다.
$$\tilde{L} = G2LAtt(G, L) = \text{softmax}\left(\frac{LG^T}{\sqrt{d_k}}\right) * G$$
- **효과**: 배경 간의 불필요한 상호작용을 제거하고 인스턴스에 필요한 전역 정보만 효율적으로 추출한다.

### 4. Proposal's Area Loss ($\text{L}_{\text{PArea}}$)

LPM과 GPM의 입력이 되는 Proposal의 품질을 높이기 위해, 위치(Location) 외에 스케일(Scale)을 분리하여 최적화하는 손실 함수를 제안한다.

- **원리**: 예측된 Proposal과 Ground Truth(GT) 박스 간의 면적 오차를 계산하여 정규화하고 평균을 낸다.
- **전체 손실 함수**:
$$\text{L}_{\text{total}} = \text{L}_{\text{PArea}} + \text{L}_{\text{box}} + \text{L}_{\text{class}} + \text{L}_{\text{mask}}$$

### 5. Promptable Instance Segmentation 구현

제안된 구조는 자동 분할뿐만 아니라, 사용자가 입력한 박스 프롬프트를 통해 특정 객체만 분할하는 기능으로 확장 가능하다. RPN의 Proposal이나 수동 입력 박스를 LPM과 GPM의 입력으로 사용하여 디코더가 마스크를 예측하도록 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ISAID(광학), NWPU VHR-10(광학), SSDD(SAR), HRSID(SAR) 총 4종의 데이터셋 사용.
- **평가 지표**: $\text{AP}, \text{AP}_{50}, \text{AP}_{75}$ 및 객체 크기별 $\text{AP}_s, \text{AP}_m, \text{AP}_l$ 사용.
- **구현**: MMdetection 프레임워크 기반, ResNet50/101 백본 사용.

### 주요 결과

1. **Ablation Study (ISAID 데이터셋)**:
    - Baseline(Mask R-CNN) 대비 LPM 추가 시 $\text{AP}_s$가 1.5% 향상.
    - GPM 추가 시 $\text{AP}_s$가 2.2% 향상.
    - $\text{L}_{\text{PArea}}$ 추가 시 $\text{AP}_s$가 1.6% 향상.
    - 모든 모듈을 통합했을 때 $\text{AP}_s$가 2.7% 향상되며 가장 높은 성능을 보였다.

2. **정량적 비교 (SOTA 대비)**:
    - **ISAID**: SCNet에 본 제안 기법을 적용했을 때 $\text{AP}=38.0\%$를 달성하여 기존 SOTA 모델들을 능가함.
    - **NWPU VHR-10**: Cascade Mask R-CNN에 적용하여 $\text{AP}=66.9\%$ 달성, 새로운 SOTA 기록.
    - **SAR 이미지 (SSDD, HRSID)**: Cascade Mask R-CNN 기반 모델이 특히 작은 객체($\text{AP}_s$)와 전반적인 성능에서 경쟁력 있는 결과를 보임.

3. **효율성 및 프롬프트 성능**:
    - 인코더 특징 추출 후, 박스 프롬프트를 통한 개별 인스턴스 분할 시간은 단 **40ms**에 불과하여 실시간 응용 가능성을 입증했다.

## 🧠 Insights & Discussion

### 강점

- **효율적인 정보 보완**: 전체 이미지를 고해상도로 처리하는 대신, 프롬프트라는 형식을 통해 필요한 부분에만 고해상도 정보를 주입함으로써 연산량 증가를 최소화하며 다운샘플링 문제를 해결했다.
- **범용성**: 특정 모델에 종속되지 않고 Mask R-CNN, Cascade Mask R-CNN, SCNet 등 다양한 기존 인스턴스 분할 모델에 모듈 형태로 결합하여 성능을 끌어올릴 수 있다.
- **다양한 센서 대응**: 광학 이미지뿐만 아니라 SAR 이미지에서도 일관된 성능 향상을 보여, 원격 탐사 데이터 일반에 효과적임을 입증했다.

### 한계 및 논의사항

- **Proposal 의존성**: LPM과 GPM의 성능이 RPN이 생성하는 Proposal의 품질에 의존한다. 이를 보완하기 위해 $\text{L}_{\text{PArea}}$를 도입했으나, Proposal 자체가 완전히 실패한 경우 프롬프트 모듈이 이를 복구하기는 어렵다.
- **하이퍼파라미터 민감도**: 프롬프트 ROI 사이즈($S_{ps}$)에 따라 성능 변화가 있으며, 본 논문에서는 28로 설정했으나 데이터셋마다 최적값이 다를 수 있다.

## 📌 TL;DR

본 논문은 원격 탐사 이미지의 고유 특성인 **낮은 전경 비율과 작은 객체 크기**, 그리고 기존 딥러닝 모델의 **다운샘플링으로 인한 정보 손실** 문제를 해결하기 위해 **LPM(국소 프롬프트), GPM(전역-국소 프롬프트), $\text{L}_{\text{PArea}}$(면적 손실 함수)**로 구성된 새로운 프롬프트 패러다임을 제안한다. 이 방법은 기존 모델의 구조를 유지하면서도 고해상도 세부 정보를 효율적으로 주입하여 4개 벤치마크 데이터셋에서 SOTA 수준의 성능을 달성했으며, 특히 40ms라는 빠른 속도로 인터랙티브한 프롬프트 기반 분할을 가능하게 함으로써 향후 원격 탐사 분석 시스템의 실용성을 크게 높일 것으로 기대된다.
