# One-shot Weakly-Supervised Segmentation in Medical Images

Wenhui Lei, Qi Su, Ran Gu, Na Wang, Xinglong Liu, Guotai Wang, Xiaofan Zhang, Shaoting Zhang (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델이 높은 성능을 내기 위해서는 정밀하고 방대한 양의 픽셀 단위 주석(pixel-level annotation)이 필요하다. 그러나 의료 영상의 특성상 도메인 전문가가 직접 주석을 달아야 하므로 시간과 비용 소모가 매우 크다. 이를 해결하기 위해 단 하나의 주석 이미지로부터 새로운 클래스를 학습하는 One-shot segmentation과 정밀한 마스크 대신 거친 라벨(coarse labels)을 사용하는 Weakly-supervised learning이 대안으로 제시되어 왔다.

기존의 One-shot segmentation(OSS) 방법론들은 주로 Prototypical networks에 기반하여 클래스 프로토타입을 추출하지만, 의료 영상에서 발생하는 두 가지 핵심 문제, 즉 (1) 작은 전경(foreground)과 거대한 배경(background) 사이의 극심한 샘플 불균형 문제와 (2) 전경과 주변 조직 간의 낮은 대비(low contrast) 문제로 인해 타겟 장기의 정확한 위치 파악 및 경계 추출에 어려움을 겪는다. 따라서 본 논문은 단 하나의 약하게 지도된(weakly labeled) 이미지와 다수의 라벨 없는 이미지를 활용하여 강건한 분할 모델을 학습시키는 프레임워크를 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 One-shot Localization(OSL), Weakly-Supervised Segmentation(WSS), 그리고 Noisy label training 전략을 유기적으로 결합하는 것이다. 

중심적인 설계 직관은 서포트 이미지의 정밀한 픽셀 마스크를 직접 생성하는 대신, 소수의 포인트(scribbles)만을 사용하여 라벨 없는 이미지들에서 타겟 위치를 찾아내고(Localization), 이를 통해 생성된 의사 마스크(pseudo masks)를 이용해 모델을 학습시키는 것이다. 특히, 인체 구조의 유사성 가설을 바탕으로 PRNet을 통해 스크리블을 전파하고, Dual-level Feature Denoising(DFD)을 통해 전파된 포인트의 정확성을 검증함으로써 노이즈를 최소화한 학습 데이터를 구축하는 것이 핵심이다.

## 📎 Related Works

기존의 OSS 연구들은 주로 메타 학습(Meta-learning)의 흐름을 따라 Metric-based, Model-based, Optimization-based 방법론으로 나뉜다. 특히 의료 영상 분야의 OSS에서는 SE-Net이나 SSL-ALPNet과 같은 방법들이 제안되었으나, 다음과 같은 한계가 존재한다. 첫째, 서포트 이미지의 프로토타입 생성에 치중하여 쿼리 이미지 자체의 내재적 정보(모양, 크기 등)를 간과하는 경향이 있다. 둘째, 대부분 2D 기반이며 추론 시 타겟 장기가 존재하는 슬라이스의 범위를 미리 지정해줘야 하는 추가적인 감독 정보가 필요하다.

또한, Self-supervised learning(SSL)을 통한 해부학적 구조 임베딩 연구들이 진행되었으나, 기존의 One-shot Localization(OSL) 방법들은 전파된 포인트가 실제로 정확한지를 스스로 확인하는 자가 검증 메커니즘이 부족하다는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 이미지 재구성(Image reconstruction) 태스크를 결합하여 해부학적 및 픽셀 레벨의 특징을 동시에 추출하고 이를 필터링에 활용한다.

## 🛠️ Methodology

본 논문이 제안하는 프레임워크는 크게 세 단계의 파이프라인으로 구성된다.

### 1. Scribble Localization (PRNet)
Scribble을 인접한 포인트들의 집합으로 간주하고, 이를 라벨 없는 이미지로 전파하기 위해 Propagation-Reconstruction Network(PRNet)를 제안한다. PRNet은 인코더, Fully Connected(FC) 레이어, 디코더로 구성되며, 두 가지 SSL 태스크를 통해 학습된다.

- **Relative Distance Regression**: 두 포인트 $c_0, c_1$ 주변의 패치를 입력받아 물리적 공간에서의 상대적 거리 $d_{10}$을 예측한다.
  $$d_{10} = (c_1 - c_0) \circ e$$
  여기서 $e$는 픽셀 간격(pixel spacing)이며, 예측값 $d'_{10}$은 다음과 같이 계산된다.
  $$d'_{10} = r \cdot \tanh(p(c_0) - p(c_1))$$
- **Image Reconstruction**: 입력 패치를 다시 재구성하여 픽셀 레벨의 특징을 학습한다.

최종 손실 함수는 거리 예측 손실($L_{dis}$)과 재구성 손실($L_{rec}$)의 합으로 정의된다.
$$L_{ssl} = L_{dis} + L_{rec} = \frac{1}{3} ||d_{10} - d'_{10}||_2^2 + \frac{1}{N} (||x(c_0) - x^r(c_0)||_2^2 + ||x(c_1) - x^r(c_1)||_2^2)$$

### 2. Dual-level Feature Denoising (DFD)
전파된 포인트 $c'_0$가 정확한지 검증하기 위해 해부학적 레벨(anatomical-level)과 픽셀 레벨(pixel-level)의 특징을 모두 활용한다. PRNet의 디코더에서 추출한 저수준 특징 맵 $m_2$와 고수준 특징 맵 $m_4$에서 각각 특징 벡터 $f_2, f_4$를 추출하여 코사인 유사도를 계산한다.
$$\text{sim} = \cos(f_2(c_0), f_2(c'_0)) \cdot \cos(f_4(c_0), f_4(c'_0))$$
이 유사도 값이 임계값 $\tau$보다 큰 경우에만 해당 포인트를 유지하며, 이를 통해 전파 과정에서 발생한 노이즈를 제거한다.

### 3. Pseudo Mask Generation and Noisy Label Training
정제된 스크리블을 바탕으로 Geodesic Image Segmentation(GeoS) 알고리즘을 적용하여 각 이미지에 대한 의사 마스크(pseudo masks)를 생성한다. 이후 3D UNet 기반의 분할 모델 $f_\theta$를 학습시키는데, 생성된 마스크에 노이즈가 포함되어 있을 수 있으므로 Progressive Label Correction(PLC) 전략을 사용한다. PLC는 모델의 예측 확신도가 임계값 $\delta$보다 높고 기존 라벨과 다를 경우, 라벨을 예측값으로 수정하며 반복적으로 학습하는 방식이다.

## 📊 Results

### 실험 설정
- **데이터셋**: TCIA(복부 장기: 간, 비장, 왼쪽 신장) 및 StructSeg19(두경부 장기: 뇌줄기, 좌우 이하선).
- **측정 지표**: Dice score (0~1 사이의 값으로 1에 가까울수록 정밀함).
- **비교 대상**: SE-Net, SSL-ALPNet, DataAug 등 기존 OSS 및 Few-shot 방법론.

### 주요 결과
실험 결과, 제안 방법은 기존 최신 방법론들보다 월등한 성능 향상을 보였다.
- **TCIA 데이터셋**: 평균 Dice score 기준 기존 방법 대비 약 23% 향상.
- **StructSeg19 데이터셋**: 평균 Dice score 기준 약 45.4% 향상.

특히, 타겟 장기의 위치 범위를 미리 알 필요가 없음에도 불구하고, 매우 낮은 대비와 극심한 클래스 불균형 상황에서도 강건한 성능을 유지하였다. 이는 기존 방법들이 타겟 장기의 정렬 상태나 슈퍼픽셀(superpixel)의 품질에 의존했던 것과 대조적이다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문의 가장 큰 강점은 단순한 프로토타입 매칭을 넘어 '위치 추정 $\rightarrow$ 특징 기반 정제 $\rightarrow$ 의사 라벨 생성 $\rightarrow$ 노이즈 학습'으로 이어지는 단계적 파이프라인을 구축했다는 점이다. 특히 DFD 모듈을 통해 해부학적 구조와 픽셀 특성을 동시에 고려함으로써, 의료 영상의 고질적인 문제인 낮은 대비 환경에서도 정확한 포인트를 선별할 수 있었다.

### 한계 및 비판적 해석
본 방법론은 "서로 다른 사람이라도 해부학적 구조가 유사하다"는 가정을 전제로 한다. 따라서 성인의 데이터로 학습된 PRNet을 영유아의 영상에 적용할 경우, 구조적 차이로 인해 위치 추정 성능이 급격히 떨어질 가능성이 높다. 저자들 역시 이를 한계점으로 명시하며, 타겟 도메인의 라벨 없는 영상으로 모델을 미세 조정(fine-tuning)하는 방법이 필요함을 언급하고 있다.

또한, GeoS를 통해 생성된 의사 마스크의 품질이 전체 성능의 하한선을 결정하므로, 향후 더 발전된 WSS 알고리즘이 도입된다면 성능을 더욱 끌어올릴 수 있을 것으로 보인다.

## 📌 TL;DR

본 연구는 단 하나의 스크리블 주석이 포함된 이미지와 다수의 라벨 없는 이미지만을 사용하여 3D 의료 영상을 분할하는 새로운 프레임워크를 제안한다. PRNet을 통한 포인트 전파, DFD를 통한 이중 레벨 특징 정제, 그리고 PLC 기반의 노이즈 라벨 학습을 결합하여, 기존 OSS 방법들이 해결하지 못한 클래스 불균형 및 저대비 문제를 효과적으로 극복하였다. 이 연구는 매우 적은 양의 주석만으로도 고성능의 의료 영상 분할 모델을 구축할 수 있는 가능성을 제시하며, 특히 장기 위치 추정과 정제 메커니즘은 향후 다양한 의료 영상 랜드마크 검출 작업에 응용될 가능성이 높다.