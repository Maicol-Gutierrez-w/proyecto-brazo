import sys
import os
import numpy as np
import trimesh

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QButtonGroup
)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt

from OpenGL.GL import *
from OpenGL.GLU import *

# Si alguna articulación gira al revés de lo esperado, cambia su signo aquí (1 o -1)
SIGNOS = {
    "gira_base": 1,
    "brazo_1": 1,
    "brazo_2": 1,
    "antebrazo": 1,
    "giro_muñeca": 1,
    "pinzas": 1,
}

PREFIJO = "Brazo ensamblado - "

GRIS = (0.25, 0.25, 0.25)
# nombre del archivo (sin prefijo ni .STL) -> color
PIEZAS = {
    "base del brazo-1": (0.85, 0.65, 0.10),
    "base 2 mano-4": (0.15, 0.35, 0.85),
    "brazo-1": (0.10, 0.65, 0.75),
    "brazo 2-2": (0.20, 0.50, 0.65),
    "antebrazo-1": (0.10, 0.75, 0.55),
    "base pinza-3": (0.85, 0.30, 0.20),
    "Pinzas-4": (0.90, 0.80, 0.15),
    "Pinzas-5": (0.95, 0.45, 0.10),
    "pin 5mm-1": GRIS, "pin 5mm-3": GRIS,
    "pin 20mm-1": GRIS, "pin 20mm-2": GRIS, "pin 20mm-3": GRIS,
}

# Vistas predefinidas: (rotacion_x, rotacion_y) en el marco ALINEADO del modelo
# (arriba = eje de la base, adelante = dirección en que el brazo se extiende en reposo).
# Ajusta aquí si quieres afinar: en "Cámara libre" la consola imprime los ángulos al soltar el mouse.
VISTAS = {
    "superior":   (75.0, 0.0),        # imagen 1: desde arriba, brazo extendiéndose hacia abajo
    "isometrica": (35.264, -135.0),   # imagen 2: desde arriba en diagonal
    "lateral":    (15.0, 90.0),       # imagen 3: de perfil (mirando a lo largo del eje del hombro)
}

# articulación -> (slider que la controla, factor)
JUNTA_SLIDER = {
    "pinza_izq": ("pinzas", 0.35),
    "pinza_der": ("pinzas", -0.35),
}


def normalizar(v, defecto=None):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None if defecto is None else np.array(defecto, dtype=float)
    return v / n


def eje_cilindro(vertices):
    """Devuelve el eje de simetría de una pieza cilíndrica (pasador) usando PCA.
    El eje es el autovector 'distinto' de los otros dos (que son casi iguales)."""
    c = vertices - vertices.mean(axis=0)
    cov = c.T @ c / len(c)
    w, v = np.linalg.eigh(cov)  # ascendente
    idx = 2 if (w[1] - w[0]) < (w[2] - w[1]) else 0
    eje = v[:, idx]
    i = np.argmax(np.abs(eje))
    if eje[i] < 0:
        eje = -eje
    return eje / np.linalg.norm(eje)


def perpendicular(v, eje):
    return v - np.dot(v, eje) * eje


class Visor3D(QOpenGLWidget):

    def __init__(self):
        super().__init__()

        self.partes = {}          # clave corta -> datos de malla
        self.grupo = {}           # clave -> nombre de articulación que la mueve (None = fija)
        self.juntas = {}          # nombre -> (pivote, eje)
        self.hijos = {}           # nombre (o None) -> [articulaciones hijas]

        # Cámara
        self.rotacion_x = 20
        self.rotacion_y = 30
        self.zoom = 1.0
        self.posicion_x = 0
        self.posicion_y = 0
        self.ultimo_mouse = None
        self.rotacion_bloqueada = False   # True en las vistas fijas
        self.mat_alineacion = np.eye(4, dtype=np.float32)  # modelo -> marco alineado (OpenGL)

        self.centro = np.zeros(3)
        self.escala = 1.0

        self.articulaciones = {
            "gira_base": 0, "brazo_1": 0, "brazo_2": 0,
            "antebrazo": 0, "giro_muñeca": 0, "pinzas": 0
        }

        self.cargar_modelos()

    # ------------------------------------------------------------------
    def cargar_modelos(self):
        carpeta = "modelos_ensamblados"
        self.partes = {}

        for nombre, color in PIEZAS.items():
            archivo = f"{PREFIJO}{nombre}.STL"
            ruta = os.path.join(carpeta, archivo)
            if not os.path.exists(ruta):
                print(f"[FALTA] {ruta}")
                continue
            try:
                m = trimesh.load(ruta, force="mesh")
                verts = np.asarray(m.vertices, dtype=np.float64)
                caras = np.asarray(m.faces)

                tri = verts[caras]
                n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                ln = np.linalg.norm(n, axis=1, keepdims=True)
                ln[ln == 0] = 1
                n = n / ln

                # nube de puntos densa para medir contactos entre piezas
                rng = np.random.default_rng(0)
                sub = verts[rng.choice(len(verts), min(len(verts), 400), replace=False)]
                try:
                    surf, _ = trimesh.sample.sample_surface(m, 400)
                    puntos = np.vstack([sub, surf])
                except Exception:
                    puntos = sub

                self.partes[nombre.lower()] = {
                    "vertices": verts,
                    "puntos": puntos,
                    "tri": np.ascontiguousarray(tri.reshape(-1, 3), dtype=np.float32),
                    "normales": np.ascontiguousarray(np.repeat(n, 3, axis=0), dtype=np.float32),
                    "centro": (verts.min(axis=0) + verts.max(axis=0)) / 2.0,
                    "color": color,
                }
            except Exception as e:
                print(f"Error al cargar {archivo}: {e}")

        try:
            self.construir_cinematica()
        except Exception as e:
            print(f"[ERROR] No se pudo deducir la cinemática: {e}")
            self.grupo = {k: None for k in self.partes}
            self.juntas, self.hijos = {}, {None: []}
        self.calcular_encuadre()

    # ------------------------------------------------------------------
    def construir_cinematica(self):
        """Deduce la jerarquía desde la geometría:
        - cada pasador une las 2 piezas cuyos agujeros lo rodean;
        - las uniones sin pasador (giro de base y de muñeca) se infieren por contacto;
        - desde la base fija hacia afuera: al mover una articulación solo se mueve
          lo que queda del lado lejano de ese pasador."""
        pernos = [k for k in self.partes if k.startswith("pin ")]
        piezas = [k for k in self.partes if not k.startswith("pin ")]
        if not piezas:
            return
        raiz = "base del brazo-1" if "base del brazo-1" in piezas else piezas[0]

        # ---- geometría de cada pasador
        info = {}
        for k in pernos:
            v = self.partes[k]["vertices"]
            c = self.partes[k]["centro"]
            e = eje_cilindro(v)
            d = v - c
            along = d @ e
            perp = np.linalg.norm(d - np.outer(along, e), axis=1)
            info[k] = {"c": c, "e": e, "r": perp.max(), "L": np.abs(along).max()}

        # ---- qué piezas atraviesa cada pasador
        pieza_de_perno = {}
        for k in pernos:
            c, e, r, L = info[k]["c"], info[k]["e"], info[k]["r"], info[k]["L"]
            puntajes = []
            for p in piezas:
                d = self.partes[p]["vertices"] - c
                along = d @ e
                perp = np.linalg.norm(d - np.outer(along, e), axis=1)
                n = np.sum((np.abs(along) <= L * 1.05) & (perp >= 0.6 * r) & (perp <= 2.0 * r))
                puntajes.append((n, p))
            puntajes.sort(reverse=True)
            mejor = puntajes[0][0]
            elegidas = [p for n, p in puntajes[:2] if n >= max(4, 0.2 * mejor)]
            pieza_de_perno[k] = elegidas
            if len(elegidas) < 2:
                print(f"[AVISO] {k}: solo se detectó {elegidas}")

        # ---- distancia mínima entre piezas (para uniones sin pasador)
        def dist_min(a, b):
            A, B = self.partes[a]["puntos"], self.partes[b]["puntos"]
            d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            return float(np.sqrt(max(d2.min(), 0.0)))

        aristas = []   # (peso, a, b, perno)
        for k, ps in pieza_de_perno.items():
            if len(ps) == 2:
                aristas.append((-1.0, ps[0], ps[1], k))
        for i, a in enumerate(piezas):
            for b in piezas[i + 1:]:
                aristas.append((dist_min(a, b), a, b, None))
        aristas.sort(key=lambda t: t[0])

        padre_uf = {p: p for p in piezas}
        def raiz_uf(x):
            while padre_uf[x] != x:
                padre_uf[x] = padre_uf[padre_uf[x]]
                x = padre_uf[x]
            return x

        arbol = {p: [] for p in piezas}
        for w, a, b, k in aristas:
            ra, rb = raiz_uf(a), raiz_uf(b)
            if ra != rb:
                padre_uf[ra] = rb
                arbol[a].append((b, k))
                arbol[b].append((a, k))

        # ---- orientar el árbol desde la base fija
        padre, prof, orden, perno_arista = {raiz: None}, {raiz: 0}, [raiz], {}
        i = 0
        while i < len(orden):
            u = orden[i]; i += 1
            for v, k in arbol[u]:
                if v not in padre:
                    padre[v] = u
                    prof[v] = prof[u] + 1
                    perno_arista[v] = k
                    orden.append(v)

        # ---- clasificar las uniones (de la base hacia afuera)
        hijas = [p for p in orden if p != raiz]
        pin20 = sorted([p for p in hijas if perno_arista[p] and "20mm" in perno_arista[p]], key=lambda p: prof[p])
        pin5 = sorted([p for p in hijas if perno_arista[p] and "5mm" in perno_arista[p]], key=lambda p: perno_arista[p])
        sinpin = sorted([p for p in hijas if not perno_arista[p]], key=lambda p: prof[p])

        nombre_union = {}
        for p, n in zip(pin20, ["brazo_1", "brazo_2", "antebrazo"]):
            nombre_union[p] = n
        for p, n in zip(sinpin, ["gira_base", "giro_muñeca"]):
            nombre_union[p] = n
        for p, n in zip(pin5, ["pinza_izq", "pinza_der"]):
            nombre_union[p] = n

        # ---- grupo de cada pieza: la articulación que la mueve (la última de su cadena)
        self.grupo = {raiz: None}
        padre_junta = {}
        for p in orden[1:]:
            j = nombre_union.get(p)
            if j is None:
                self.grupo[p] = self.grupo[padre[p]]       # unión rígida
            else:
                self.grupo[p] = j
                padre_junta[j] = self.grupo[padre[p]]
        for k, ps in pieza_de_perno.items():
            if not ps:
                self.grupo[k] = None
                continue
            hijo = max(ps, key=lambda p: prof.get(p, 0))
            self.grupo[k] = self.grupo[hijo]               # el pasador viaja con el lado lejano

        self.hijos = {None: []}
        for j in padre_junta:
            self.hijos.setdefault(j, [])
        for j, pj in padre_junta.items():
            self.hijos.setdefault(pj, []).append(j)

        # ---- pivote y eje de cada articulación
        def pieza_de(j):
            return next((p for p, n in nombre_union.items() if n == j), None)

        self.juntas = {}
        for p, j in nombre_union.items():
            k = perno_arista.get(p)
            if k:
                self.juntas[j] = (info[k]["c"], info[k]["e"])

        def contacto(a, b):
            A, B = self.partes[a]["puntos"], self.partes[b]["puntos"]
            d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            dm = np.sqrt(np.maximum(d2.min(axis=1), 0))
            tol = dm.min() + 0.03 * np.linalg.norm(A.max(0) - A.min(0))
            return A[dm <= tol].mean(axis=0)

        # giro de la base: eje perpendicular al hombro, desde la base hacia el hombro
        pb = pieza_de("gira_base")
        if pb:
            if "brazo_1" in self.juntas:
                c1, e1 = self.juntas["brazo_1"]
                e = normalizar(perpendicular(c1 - self.partes[raiz]["centro"], e1))
            else:
                e = None
            if e is None:
                e = np.array([0.0, 1.0, 0.0])
            self.juntas["gira_base"] = (contacto(padre[pb], pb), e)

        # giro de la muñeca (roll): el eje va a lo largo del eslabón donde está la unión,
        # perpendicular a los pasadores vecinos. Se calcula con el pasador anterior (lado
        # de la base) y/o el siguiente (lado de las pinzas), sin importar en qué orden estén.
        pm = pieza_de("giro_muñeca")
        if pm:
            pp = padre[pm]
            centro_union = contacto(pp, pm)
            pj = self.grupo.get(pp)
            prev = self.juntas.get(pj) if pj and pj not in ("gira_base", "giro_muñeca") else None
            nxt = None
            for h in self.hijos.get(self.grupo.get(pm), []):
                if h in self.juntas and not h.startswith("pinza") and h != "giro_muñeca":
                    nxt = self.juntas[h]
                    break

            e, piv = None, centro_union
            if prev is not None and nxt is not None:
                e = normalizar(perpendicular(nxt[0] - prev[0], prev[1])); piv = prev[0]
            elif prev is not None:
                e = normalizar(perpendicular(centro_union - prev[0], prev[1])); piv = prev[0]
            elif nxt is not None:
                e = normalizar(perpendicular(nxt[0] - centro_union, nxt[1])); piv = nxt[0]
            if e is None:
                e = normalizar(self.partes[pm]["centro"] - centro_union, [0, 1, 0])
            self.juntas["giro_muñeca"] = (piv, e)

        # lo que no se pudo detectar queda rígido (no se mueve)
        self.hijos = {k: [j for j in v if j in self.juntas] for k, v in self.hijos.items()}
        self.hijos.setdefault(None, [])

        self.calcular_alineacion(raiz)

        print("=== Jerarquía deducida (base -> pinzas) ===")
        def imprimir(j, nivel=0):
            ps = [p for p, g in self.grupo.items() if g == j]
            if j is None:
                print("FIJO:", ps)
            else:
                piv, eje = self.juntas[j]
                print("  " * nivel + f"{j}: mueve {ps} | pivote={np.round(piv,1)} eje={np.round(eje,3)}")
            for h in self.hijos.get(j, []):
                imprimir(h, nivel + 1)
        imprimir(None)

    def calcular_alineacion(self, raiz):
        """Gira el modelo para que 'arriba' sea el eje de la base y 'adelante' la dirección
        en que el brazo se extiende en reposo. Así las vistas no dependen de cómo
        exportó SolidWorks los ejes."""
        self.mat_alineacion = np.eye(4, dtype=np.float32)
        try:
            if "gira_base" not in self.juntas or "brazo_1" not in self.juntas:
                print("[AVISO] No se pudo alinear el modelo; se usan los ejes originales.")
                return
            u = self.juntas["gira_base"][1]
            s_ = perpendicular(self.juntas["brazo_1"][1], u)
            s_ = normalizar(s_)
            if s_ is None:
                return
            f = normalizar(np.cross(u, s_))

            # sentido de 'adelante': hacia donde apuntan las pinzas en reposo
            pinzas = [self.partes[k]["centro"] for k in self.partes if k.startswith("pinzas")]
            if pinzas:
                d = np.mean(pinzas, axis=0) - self.juntas["brazo_1"][0]
                d = perpendicular(d, u)
                if np.linalg.norm(d) > 1e-6 and np.dot(d, f) < 0:
                    f = -f

            x_ = np.cross(u, f)                      # derecha = arriba x adelante
            R = np.vstack([x_, u, f])                # filas: x', y', z'
            R4 = np.eye(4)
            R4[:3, :3] = R
            self.mat_alineacion = np.ascontiguousarray(R4.T, dtype=np.float32)  # column-major
            print(f"Alineación: arriba={np.round(u,3)} adelante={np.round(f,3)} lateral={np.round(x_,3)}")
        except Exception as e:
            print(f"[AVISO] Alineación omitida: {e}")

    def calcular_encuadre(self):
        if not self.partes:
            return
        todos = np.vstack([m["vertices"] for m in self.partes.values()])
        minimo, maximo = todos.min(axis=0), todos.max(axis=0)
        self.centro = (minimo + maximo) / 2.0
        mayor = max(maximo - minimo)
        if mayor > 0:
            self.escala = 4.0 / mayor

    # ------------------------------------------------------------------
    def initializeGL(self):
        glClearColor(0.08, 0.09, 0.12, 1.0)
        glEnable(GL_DEPTH_TEST)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.35, 0.35, 0.35, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.85, 0.85, 0.85, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    def resizeGL(self, width, height):
        if height == 0:
            height = 1
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, width / height, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def angulo_junta(self, j):
        slider, factor = JUNTA_SLIDER.get(j, (j, 1.0))
        return self.articulaciones[slider] * SIGNOS[slider] * factor

    def dibujar_junta(self, j):
        """Aplica la rotación de j (sobre su pivote), dibuja solo lo que ella mueve
        y anida las articulaciones siguientes. Lo que está del lado de la base no se toca."""
        pivote, eje = self.juntas[j]
        glPushMatrix()
        glTranslatef(*pivote)
        glRotatef(self.angulo_junta(j), *eje)
        glTranslatef(*(-pivote))
        for k, g in self.grupo.items():
            if g == j:
                self.dibujar_pieza(k)
        for h in self.hijos.get(j, []):
            self.dibujar_junta(h)
        glPopMatrix()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, [0.4, 0.7, 1.0, 0.0])

        glTranslatef(self.posicion_x, self.posicion_y, -8.0 * self.zoom)
        glRotatef(self.rotacion_x, 1, 0, 0)
        glRotatef(self.rotacion_y, 0, 1, 0)
        glScalef(self.escala, self.escala, self.escala)
        glMultMatrixf(self.mat_alineacion)
        glTranslatef(*(-self.centro))

        for k, g in self.grupo.items():
            if g is None:
                self.dibujar_pieza(k)
        for j in self.hijos.get(None, []):
            self.dibujar_junta(j)

    def dibujar_pieza(self, clave):
        m = self.partes.get(clave)
        if m is None:
            return
        glColor3f(*m["color"])
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, m["tri"])
        glNormalPointer(GL_FLOAT, 0, m["normales"])
        glDrawArrays(GL_TRIANGLES, 0, len(m["tri"]))
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        self.ultimo_mouse = event.position()

    def mouseMoveEvent(self, event):
        if self.ultimo_mouse is None:
            return
        pos = event.position()
        dx = pos.x() - self.ultimo_mouse.x()
        dy = pos.y() - self.ultimo_mouse.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            if not self.rotacion_bloqueada:
                self.rotacion_y += dx * 0.5
                self.rotacion_x += dy * 0.5
        elif event.buttons() & Qt.MouseButton.RightButton:
            self.posicion_x += dx * 0.01
            self.posicion_y -= dy * 0.01

        self.ultimo_mouse = pos
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.rotacion_bloqueada:
            print(f"Cámara: rotacion_x={self.rotacion_x:.1f}  rotacion_y={self.rotacion_y:.1f}")

    def wheelEvent(self, event):
        self.zoom *= 0.9 if event.angleDelta().y() > 0 else 1.1
        self.zoom = max(0.1, min(10.0, self.zoom))
        self.update()

    def poner_vista(self, rot_x, rot_y, bloquear=True):
        """Coloca la cámara en una vista predefinida (zoom y desplazamiento al inicio)."""
        self.rotacion_x = rot_x
        self.rotacion_y = rot_y
        self.zoom = 1.0
        self.posicion_x = 0
        self.posicion_y = 0
        self.rotacion_bloqueada = bloquear
        self.update()

    def vista_superior(self):
        self.poner_vista(*VISTAS["superior"])

    def vista_lateral(self):
        self.poner_vista(*VISTAS["lateral"])

    def vista_isometrica(self):
        self.poner_vista(*VISTAS["isometrica"])

    def camara_libre(self):
        self.rotacion_bloqueada = False   # conserva la vista actual y permite rotar

    def reiniciar_camara(self):
        self.rotacion_x = 20
        self.rotacion_y = 30
        self.zoom = 1.0
        self.posicion_x = 0
        self.posicion_y = 0
        self.update()


class VentanaPrincipal(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador 3D - Brazo Robótico Articulado")
        self.resize(1300, 800)
        self.crear_interfaz()

    def crear_interfaz(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout_principal = QHBoxLayout()
        central.setLayout(layout_principal)

        panel = QVBoxLayout()

        titulo = QLabel("CONTROL DE MOVIMIENTO")
        titulo.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px;")
        panel.addWidget(titulo)

        self.sliders = {}
        controles = [
            ("Giro Base", "gira_base", -180, 180, 0),
            ("Brazo Principal", "brazo_1", -90, 90, 0),
            ("Brazo Secundario", "brazo_2", -90, 90, 0),
            ("Antebrazo", "antebrazo", -90, 90, 0),
            ("Giro Muñeca", "giro_muñeca", -180, 180, 0),
            ("Apertura Pinzas", "pinzas", 0, 100, 0)
        ]

        for nombre, clave, min_val, max_val, default in controles:
            unidad = "%" if clave == "pinzas" else "°"
            lbl = QLabel(f"{nombre}: {default}{unidad}")
            lbl.setStyleSheet("font-weight: bold; color: #a0c0d0; margin-top: 5px;")
            panel.addWidget(lbl)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            slider.valueChanged.connect(
                lambda val, k=clave, l=lbl, n=nombre: self.actualizar_articulacion(k, val, l, n)
            )
            panel.addWidget(slider)
            self.sliders[clave] = (slider, lbl, nombre)

        self.visor = Visor3D()

        etiqueta_vistas = QLabel("VISTAS DE CÁMARA")
        etiqueta_vistas.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 15px;")
        panel.addWidget(etiqueta_vistas)

        self.grupo_vistas = QButtonGroup(self)
        self.grupo_vistas.setExclusive(True)
        vistas = [
            ("Cámara libre", self.visor.camara_libre),
            ("Vista superior", self.visor.vista_superior),
            ("Vista isométrica", self.visor.vista_isometrica),
            ("Vista lateral", self.visor.vista_lateral),
        ]
        self.botones_vista = {}
        for texto, accion in vistas:
            b = QPushButton(texto)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, f=accion: f())
            self.grupo_vistas.addButton(b)
            panel.addWidget(b)
            self.botones_vista[texto] = b
        self.botones_vista["Cámara libre"].setChecked(True)

        boton_home = QPushButton("Regresar Cámara")
        boton_home.clicked.connect(self.reiniciar)
        panel.addWidget(boton_home)
        panel.addStretch()

        layout_principal.addLayout(panel, 1)
        layout_principal.addWidget(self.visor, 3)

        self.setStyleSheet("""
            QMainWindow { background-color: #1a1c23; }
            QLabel { color: white; }
            QPushButton {
                background-color: #087f91; color: white; padding: 10px;
                border-radius: 4px; font-weight: bold; margin-top: 10px;
            }
            QPushButton:hover { background-color: #10a5ba; }
            QPushButton:checked { background-color: #e0a010; color: #101010; }
            QSlider::groove:horizontal {
                border: 1px solid #444; height: 6px;
                background: #2a2d37; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #087f91; width: 14px; margin: -4px 0; border-radius: 7px;
            }
            QSlider::handle:horizontal:hover { background: #10a5ba; }
        """)

    def actualizar_articulacion(self, clave, valor, label, nombre):
        unidad = "%" if clave == "pinzas" else "°"
        label.setText(f"{nombre}: {valor}{unidad}")
        self.visor.articulaciones[clave] = valor
        self.visor.update()

    def reiniciar(self):
        self.visor.reiniciar_camara()
        self.visor.rotacion_bloqueada = False
        self.botones_vista["Cámara libre"].setChecked(True)


def main():
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()