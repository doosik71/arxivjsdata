# RASNet: Refined Attention Segmentation Network를 이용한 수술 영상 내 수술 도구 추적을 위한 분할

Zhen-Liang Ni, Gui-Bin Bian, Xiao-Liang Xie, Zeng-Guang Hou, Xiao-Hu Zhou, Yan-Jie Zhou

## 🧩 Problem to Solve

- 로봇 보조 수술에서 수술 도구의 정확한 공간 정보를 획득하고 추적하기 위한 정밀한 분할(segmentation) 및 카테고리 식별이 중요합니다.
- 기존 연구들은 주로 수술 도구의 바운딩 박스(bounding box)만을 제공하여 정밀한 경계를 얻기 어렵다는 한계가 있었습니다.
- 수술 도구 분할 과정에서 다음과 같은 도전 과제들이 발생합니다:
  - 좁은 시야각으로 인해 수술 도구의 일부만 보이는 경우.
  - 수술 도구의 자세 변화에 따라 형상이 달라지는 문제.
  - 수술 도구가 영상 전체에서 차지하는 비율이 작아 전경 픽셀(foreground pixels)이 배경 픽셀(background pixels)보다 현저히 적어 발생하는 심각한 클래스 불균형(class imbalance) 문제.

## ✨ Key Contributions

- **RASNet(Refined Attention Segmentation Network) 제안:** 수술 도구를 동시에 분할하고 해당 카테고리를 식별하는 새로운 네트워크를 제안했습니다.
- **Attention Fusion Module (AFM) 도입:** 네트워크가 핵심 영역에 집중하도록 돕는 어텐션(attention) 모듈을 설계하여 분할 정확도를 크게 향상시켰습니다.
- **가중 손실 함수 활용:** 클래스 불균형 문제를 해결하고 작은 객체에 대한 분할 성능을 개선하기 위해 크로스 엔트로피(cross entropy) 손실과 Jaccard 지수(Jaccard index)의 로그값의 가중 합을 손실 함수로 사용했습니다.
- **전이 학습(Transfer Learning) 적용:** ImageNet으로 사전 훈련된 ResNet-50을 인코더로 활용하여 네트워크의 특징 추출 능력을 강화하고 분할 정확도를 높였습니다.
- **최신 성능 달성:** MICCAI EndoVis Challenge 2017 데이터셋에서 94.65%의 평균 Dice 계수와 90.33%의 평균 IOU(Intersection Over Union)를 달성하여 해당 분야의 최신 기술(state-of-the-art) 성능을 보여주었습니다.

## 📎 Related Works

- **수술 도구 감지 및 추적:**
  - Bareum et al. [4]: YOLO를 수정한 실시간 수술 도구 추적 (바운딩 박스 제공).
  - Amy et al. [5]: Faster R-CNN을 이용한 수술 도구 분류 및 바운딩 박스 회귀.
  - Duygu et al. [1]: Region Proposal Network와 투 스트림(two-stream) 컨볼루션 네트워크를 병합하여 수술 도구 감지.
  - Iro et al. [6]: CSL 모델을 제안하여 수술 도구의 동시 분할 및 자세 추정 수행 (본 연구의 영감).
- **의료 영상 분할 네트워크:**
  - U-Net [10]: 의료 영상 분할 분야에서 널리 사용되며 뛰어난 성능을 보인 U-형 네트워크.
  - TernausNet [11]: ImageNet으로 사전 훈련된 VGG11 인코더를 사용하는 U-Net 변형.
- **어텐션 메커니즘:**
  - Semantic Segmentation을 위한 판별 특징 네트워크 [7], 피라미드 어텐션 네트워크 [8], Squeeze-and-Excitation Networks [9] 등.
- **손실 함수:**
  - Jaccard 지수를 활용한 손실 함수 [11].

## 🛠️ Methodology

- **네트워크 아키텍처 (RASNet):**
  - **U-형 네트워크:** 심층 의미 특징을 캡처하는 수축 경로(인코더)와 정밀한 위치 파악을 위한 확장 경로(디코더)로 구성됩니다.
  - **인코더:** ImageNet으로 사전 훈련된 ResNet-50 [12]을 사용하며, 7x7 컨볼루션 레이어(스트라이드 2)와 3x3 최대 풀링 레이어(스트라이드 2)로 시작하여 4개의 인코더 블록을 포함합니다.
  - **디코더:** 정보 손실을 줄이기 위해 디컨볼루션(deconvolution)을 사용하여 업샘플링을 수행합니다.
  - **스킵 연결(Skip Connections):** 고수준 특징 맵과 저수준 특징 맵을 병합하여 상세 정보와 의미 정보를 결합합니다.
  - **출력:** 원본 이미지와 동일한 크기의 분할 마스크를 생성합니다.
- **Attention Fusion Module (AFM):**
  - U-Net의 단순한 특징 연결 방식을 개선하기 위해 설계되었습니다.
  - 고수준 특징의 전역 컨텍스트(global context)를 추출하기 위해 전역 평균 풀링(global average pooling)을 수행합니다.
  - 1x1 컨볼루션과 배치 정규화(batch normalization)를 통해 가중치를 정규화하고, 소프트맥스(softmax) 함수를 사용하여 가중치 합이 1이 되도록 합니다.
  - 저수준 특징에 이 가중치 벡터를 곱하고, 가중치가 적용된 저수준 특징을 고수준 특징에 더하여 다단계 특징을 효과적으로 융합합니다. 이는 고수준 특징의 전역 컨텍스트를 저수준 특징의 정밀한 위치 정보 선택에 활용합니다.
- **Decoder Block:**
  - 특징 맵의 차원을 줄이기 위해 1x1 컨볼루션과 배치 정규화를 수행합니다.
  - 4x4 디컨볼루션(스트라이드 2)과 배치 정규화를 통해 특징 맵을 업샘플링합니다.
  - 마지막으로 1x1 컨볼루션과 배치 정규화를 통해 특징 맵의 차원을 조정합니다.
- **손실 함수:**
  - 크로스 엔트로피 손실($H$)과 Jaccard 지수($J$)의 로그값의 가중 합($L$)을 사용합니다.
    $$L = H - \alpha \log(J)$$
  - **크로스 엔트로피 손실 ($H$):** 픽셀 단위 분류에 사용됩니다.
    $$H = -\frac{1}{w \times h} \sum_{k=1}^{c} \sum_{i=1}^{w} \sum_{j=1}^{h} y_{ijk} \log \left( \frac{e^{\hat{y}_{ijk}}}{\sum_{k=1}^{c} e^{\hat{y}_{ijk}}} \right)$$
    여기서 $w, h$는 예측의 너비와 높이, $c$는 클래스 수, $y_{ijk}$는 픽셀의 정답, $\hat{y}_{ijk}$는 픽셀의 예측값입니다.
  - **Jaccard 지수 ($J$):** 클래스 불균형 문제를 완화하고 작은 객체 분할에 유리합니다.
    $$J = \frac{TP}{TP + FP + FN}$$
    여기서 $TP$는 참 양성(True Positives), $FP$는 거짓 양성(False Positives), $FN$은 거짓 음성(False Negatives) 픽셀의 부분집합을 나타냅니다.
  - $\alpha$: 크로스 엔트로피 손실과 Jaccard 지수 로그값을 균형 잡는 가중치로, 실험을 통해 0.3으로 설정되었습니다.

## 📊 Results

- **데이터셋:** MICCAI EndoVis Challenge 2017의 트레이닝 세트를 활용했습니다 (총 1800장 이미지, 1920x1080 해상도). 이 중 1400장을 훈련 세트로, 400장을 테스트 세트로 사용했습니다.
- **전이 학습 효과:**
  - 무작위 초기화된 RASNet은 평균 Dice 78.11%, 평균 IOU 70.72%를 달성했습니다.
  - ImageNet으로 사전 훈련된 RASNet은 평균 Dice **94.65%**, 평균 IOU **90.33%**를 달성하여 사전 훈련이 분할 정확도를 크게 향상시킴을 입증했습니다.
- **AFM(Attention Fusion Module) 효과:**
  - AFM이 없는 RASNet은 평균 Dice 89.31%, 평균 IOU 82.75%를 기록했습니다.
  - AFM을 포함한 RASNet은 평균 Dice **94.65%**, 평균 IOU **90.33%**를 달성하여 AFM 사용으로 Dice 5.34%, IOU 7.58%의 성능 향상을 가져왔습니다.
- **다른 네트워크와의 비교:**
  - U-Net은 평균 Dice 70.04%, 평균 IOU 56.76%를 기록했습니다.
  - TernausNet은 평균 Dice 88.04%, 평균 IOU 80.34%를 기록했습니다.
  - RASNet은 이들 네트워크보다 월등히 우수한 성능을 보이며 수술 도구 분할에서 최신 성능을 달성했습니다.
- **클래스별 성능:** Curved Scissors가 99.01%의 Dice 계수로 가장 높은 성능을 보였습니다. 반면 Prograsp Forceps (84.59%)와 Grasping Retractor (88.05%)는 상대적으로 낮은 성능을 기록했습니다.

## 🧠 Insights & Discussion

- **RASNet의 우수한 성능 원인:**
  1. ImageNet으로 사전 훈련된 ResNet-50 인코더의 강력한 특징 추출 능력.
  2. Attention Fusion Module(AFM)이 고수준 특징의 전역 컨텍스트를 활용하여 저수준 특징의 정밀한 위치 정보를 선택하도록 효과적으로 안내.
  3. 클래스 불균형 문제를 성공적으로 해결하고 작은 객체 분할 성능을 개선한 가중 손실 함수 (크로스 엔트로피 + Jaccard 지수 로그값) 사용.
- **성능 저하 사례 분석 및 한계:**
  - Prograsp Forceps의 낮은 성능은 Bipolar Forceps와 형태가 매우 유사하여 오분류되는 경우가 많기 때문입니다.
  - Grasping Retractor의 낮은 성능은 데이터셋 내 샘플 수가 매우 적어 네트워크의 과소적합(underfitting)으로 이어진 것으로 분석됩니다.
- **향후 연구 방향:** 현재의 성과를 바탕으로 보다 효과적인 어텐션 모듈 설계에 집중하여 네트워크 성능을 더욱 향상시킬 계획입니다.

## 📌 TL;DR

로봇 수술에서 수술 도구의 정밀한 분할 및 식별은 핵심적이지만, 기존 방식은 경계 정확도와 클래스 불균형 문제에 취약했습니다. 본 논문은 이를 해결하기 위해 **RASNet(Refined Attention Segmentation Network)**을 제안합니다. RASNet은 ImageNet으로 사전 훈련된 ResNet-50 인코더를 활용하며, 고수준 특징의 전역 컨텍스트를 통해 저수준 특징의 정밀한 위치 정보를 가이드하는 **Attention Fusion Module(AFM)**을 통합했습니다. 또한, 클래스 불균형에 강건한 **크로스 엔트로피와 Jaccard 지수 로그값의 가중 합 손실 함수**를 사용했습니다. 실험 결과, RASNet은 MICCAI EndoVis Challenge 2017 데이터셋에서 평균 Dice 94.65% 및 평균 IOU 90.33%를 달성하여 기존 방법들을 크게 능가하는 **최신(state-of-the-art) 성능**을 입증했습니다. 특히 AFM과 전이 학습이 성능 향상에 결정적인 역할을 했습니다.
