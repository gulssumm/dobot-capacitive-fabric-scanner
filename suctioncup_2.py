from pydobotplus import Dobot, CustomPosition
import time

time.sleep(3)

device = Dobot(port="COM3")
home_pose = device.get_pose()

# Pick pozisyonu üstten yaklaş
pos1_above = CustomPosition(
    x=home_pose.position.x - 10,
    y=home_pose.position.y + 10,
    z=5,      
    r=home_pose.position.r
)

# Pick pozisyonu tam (Birinci kutuyu almak için)
pos1 = CustomPosition(
    x=home_pose.position.x - 10,
    y=home_pose.position.y + 10,
    z=-45,    
    r=home_pose.position.r
)

# Place pozisyon üstten yaklaş (Birinci kutuyu yerleştireceğiz)
pos2_above = CustomPosition(
    x=home_pose.position.x - 100,
    y=home_pose.position.y - 100,
    z=5,        
    r=home_pose.position.r
)

# Place pozisyon tam (Birinci kutuyu yerleştireceğiz)
pos2 = CustomPosition(
    x=home_pose.position.x - 100,
    y=home_pose.position.y - 100,
    z=-30,
    r=home_pose.position.r
)

# İkinci kutuyu almak için pozisyonlar
pos3_above = CustomPosition(
    x=home_pose.position.x + 60,  # İkinci kutu için farklı X ve Y değeri
    y=home_pose.position.y + 60,
    z=50,  # Kutuyu bırakırken daha yüksek Z değeri
    r=home_pose.position.r
)

pos3 = CustomPosition(
    x=home_pose.position.x + 60,  # İkinci kutu için farklı X ve Y değeri
    y=home_pose.position.y + 60,
    z=-65,  # İkinci kutuyu almak için daha aşağıya Z değeri
    r=home_pose.position.r
)

# Pick hareketi - Birinci kutuyu alıyoruz
device.move_to(position=pos1_above, wait=True)  # Üstten yaklaşma
device.move_to(position=pos1, wait=True)  # Birinci kutuyu almak için Z'yi düşürüyoruz
time.sleep(0.5)

device.suck(True)  # Emişi aç
time.sleep(2)  # Yeterince tutması için bekliyoruz

# Birinci kutuyu yerleştirme hareketi
device.move_to(position=pos1_above, wait=True)  # Birinci kutuyu aldıktan sonra yukarıya kaldırıyoruz

# Place hareketi - Birinci kutuyu yerleştiriyoruz
device.move_to(position=pos2_above, wait=True)
time.sleep(0.5)
device.move_to(position=pos2, wait=True)
time.sleep(0.5)
device.suck(False)  # Emişi kapatıyoruz
time.sleep(0.5)
device.move_to(position=pos2_above, wait=True)

# İkinci kutuyu almak için hareket
device.move_to(position=pos3_above, wait=True)  # İkinci kutuya üstten yaklaşma
device.move_to(position=pos3, wait=True)  # İkinci kutuyu almak için Z'yi düşürüyoruz
time.sleep(0.5)

device.suck(True)  # Emişi aç
time.sleep(2)  # Yeterince tutması için bekliyoruz

# İkinci kutuyu yerleştirme hareketi
device.move_to(position=pos3_above, wait=True)  # İkinci kutuyu aldıktan sonra yukarıya kaldırıyoruz

# Place hareketi - İkinci kutuyu yerleştiriyoruz
pos2_above_2 = CustomPosition(  # İkinci kutuyu ilk kutunun hemen yanına yerleştirmek için
    x=home_pose.position.x - 40,  # X değerini biraz daha küçük yapıyoruz, böylece kutular yan yana olur
    y=home_pose.position.y - 100,
    z=-5,  # Daha yüksek Z değeri ile bırakıyoruz
    r=home_pose.position.r
)

pos2_2 = CustomPosition(  # İkinci kutuyu yerleştirme pozisyonu
    x=home_pose.position.x - 40,  # X değeri daha küçük olacak
    y=home_pose.position.y - 100,
    z=-40,  # Z değeri aynı kalıyor
    r=home_pose.position.r
)

device.move_to(position=pos2_above_2, wait=True)  # İkinci kutuya yeni yerleştirme pozisyonu
time.sleep(0.5)
device.move_to(position=pos2_2, wait=True)
time.sleep(0.5)
device.suck(False)  
time.sleep(0.5)
device.move_to(position=pos2_above_2, wait=True)  # İkinci kutuyu yerleştirdikten sonra yukarıya kaldırıyoruz

# Robotu başlangıç pozisyonuna geri getiriyoruz
device.move_to(position=home_pose.position, wait=True)

device.close()
