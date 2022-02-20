package main

import (
	"math"
	"os"
	"C"
	"fmt"
	"encoding/csv"
	"io/ioutil"
	"log"


	"github.com/golang/geo/r3"
	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	common "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
	events "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

// MAJOIRTY OF CODE IS WRITTEN BY 87andrewh, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go
// MAJOIRTY OF CODE IS WRITTEN BY 87andrewh, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go





// USEAGE go run main.go "path_to_demos" -> outputs data to /data






// Defines amount of frames to collect around attacks
const samplesPerSecond = 32
const secondsBeforeAttack = 5
const secondsAfterAttack = 1
const secondsPerAttack = secondsBeforeAttack + secondsAfterAttack
const samplesPerAttack = int(samplesPerSecond * secondsPerAttack)

// PlayerData stores all data of a player in a single frame.
type PlayerData struct {
    ViewDirectionY float32
    ViewDirectionX float32
    ammoleft  int
    nSpotted  int
    score     int
    Ping      int
    IsScoped  bool
	flashleft float32
	id 		  uint64
	name      string
    xv        float64
    yv        float64
    zv        float64
	weapon    int
	position  r3.Vector
	yaw       float32
	pitch     float32
	crouching bool
	firing    bool
	health    int
}

// AttackTime marks when an attacker shot a victim
type AttackTime struct {
	attacker    int
	victim      int
	startFrame  int
	attackFrame int
	endFrame    int
	a_spotted_v  bool
	v_spotted_a  bool
}

// FireFrameKey is a key to a dictionary that marks
// if a shooter shoot at a given frame
type FireFrameKey struct {
	shooter int
	frame   int
}

// AttackData stores the features of a single sample fed into the model.
type AttackData struct {
	// Whether the attacker used an aimbot during the attack
	attackerAimbot bool

	// One-hot encoding of attacking gun
	weaponAK47 bool
	weaponM4A4 bool
	weaponAWP  bool

	// Viewangle deltas
	attackerDeltaYaw   [samplesPerAttack]float32
	attackerDeltaPitch [samplesPerAttack]float32
    victimDeltaYaw   [samplesPerAttack]float32
	victimDeltaPitch [samplesPerAttack]float32

	// Angles between the attacker's crosshair and the victim
	crosshairToVictimYaw   [samplesPerAttack]float32
	crosshairToVictimPitch [samplesPerAttack]float32

    crosshairToAttackerYaw   [samplesPerAttack]float32
	crosshairToAttackerPitch [samplesPerAttack]float32

	victimDistance    [samplesPerAttack]float32
	attackerCrouching [samplesPerAttack]bool
	victimCrouching   [samplesPerAttack]bool
	attackerFiring    [samplesPerAttack]bool
    victimFiring    [samplesPerAttack]bool

	attackerHealth [samplesPerAttack]int
	victimHealth   [samplesPerAttack]int

	attackerViewVector     [samplesPerAttack]r3.Vector
	attackerToVictimVector [samplesPerAttack]r3.Vector

	attackerX [samplesPerAttack]float32
	attackerY [samplesPerAttack]float32
	attackerZ [samplesPerAttack]float32

	victimX [samplesPerAttack]float32
	victimY [samplesPerAttack]float32
	victimZ [samplesPerAttack]float32

	flashleft [samplesPerAttack]float32
	id [samplesPerAttack]uint64

	attackerxv      [samplesPerAttack]float64
	attackeryv      [samplesPerAttack]float64
	attackerzv      [samplesPerAttack]float64

	victimxv      [samplesPerAttack]float64
	victimyv      [samplesPerAttack]float64
	victimzv      [samplesPerAttack]float64

    a_spotted_v  [samplesPerAttack]bool
	v_spotted_a  [samplesPerAttack]bool

	name      [samplesPerAttack]string
	visible   [samplesPerAttack]bool
	frame [samplesPerAttack]int
	IsScoped [samplesPerAttack]bool
    Ping [samplesPerAttack]int

    weapon [samplesPerAttack]int
    nSpotted [samplesPerAttack]int
	filename string
	AmmoLeft [samplesPerAttack]int
	ViewDirectionY [samplesPerAttack]float32
    ViewDirectionX [samplesPerAttack]float32
    enemyPing [samplesPerAttack]int
}

// Marks guns that the model will be trained on
// TODO: Test model on different sets of guns.
var validGuns = map[string]bool{
	"AK-47":  true,
	"M4A4":   true,
	"AWP":    true,
	"M4A1":   true,
	"AUG":    true,
	"SG 553": true,
}

// Stores data to be fed into model
var modelData = []AttackData{}


//export startparsing
func startparsing(){
    source_go := os.Args[1]
    fmt.Println(source_go)
	files, err := ioutil.ReadDir(source_go)

	if err != nil {
		log.Fatal(err)
	}
	for _, f := range files {
		parseDemo(source_go, f.Name())
	}
	csvExport()
}

func main(){
    startparsing()
}


func parseDemo(source string, name string) {
    var suspect = uint64(42)
	// Times when a player is attacked by a valid gun
	var attackTimes = []AttackTime{}
	// Marks if a player is firing at a given frame.
	var fireFrames = map[FireFrameKey]bool{}
	// Marks frames surrounding attacks that should be gathered
	// for easier processing into model inputs
	var isMarked = map[int]bool{}
	// Stores the PlayerData for each player for each marked framed
	var markedFrameData = map[int]map[int]PlayerData{}
	//var markedFrameData = make([]int, 0)
	// Marks if the demo was generated with an aimbot

	f, err := os.Open(source + name)
	defer f.Close()
	checkError(err)
	p := dem.NewParser(f)

	h, err := p.ParseHeader()
	FrameRate := h.FrameRate()

	// Calculate the demo framerate with some hacks
	tick := -1
	for !(2900 < tick && tick < 3000) {
		_, err = p.ParseNextFrame()
		tick = p.GameState().IngameTick()
	}
	checkError(err)
	iters := 10
	for i := 0; i < iters; i++ {
		_, err = p.ParseNextFrame()
		checkError(err)
	}
	nextTick := p.GameState().IngameTick()

	TicksPerFrame := float64(nextTick-tick) / float64(iters)
	FrameRate2 := p.TickRate() / TicksPerFrame

	if FrameRate == 0 {
		FrameRate = FrameRate2
	}

	var framesBeforeAttack int
	var framesAfterAttack int
	if (math.Abs(FrameRate-32.0) < 1) && (FrameRate2 == 32) {
		framesBeforeAttack = secondsBeforeAttack * 32
		framesAfterAttack = secondsAfterAttack * 32
	} else if (math.Abs(FrameRate-64.0) < 4) && (FrameRate2 == 64) {
		framesBeforeAttack = secondsBeforeAttack * 64
		framesAfterAttack = secondsAfterAttack * 64
	} else if (math.Abs(FrameRate-128) < 4) && (FrameRate2 == 128) {
		framesBeforeAttack = secondsBeforeAttack * 128
		framesAfterAttack = secondsAfterAttack * 128
	} else {
		println("Invalid frame rate: ", FrameRate, FrameRate2)
		return
	}

	framesPerAttack := framesBeforeAttack + framesAfterAttack
	framesPerSample := int(framesPerAttack / samplesPerAttack)
	//println("Frames per sample ", framesPerSample)

	// First pass.

	// Get frame times of attacks with valid guns,
	// and mark surrounding frames for retrieval.
	var attackCount int = 0
	var start int = 0
	var end int = 0
	var frame int = 0
	var attackFrame int

	p.RegisterEventHandler(func(e events.PlayerHurt) {
		 if e.Attacker.SteamID64 != suspect{
		    if suspect != uint64(42){
			    return
		    }
		}
		attackCount++
		attackFrame = p.CurrentFrame()
		start = attackFrame - framesBeforeAttack
		end = attackFrame + framesAfterAttack
		for frame = start; frame < end; frame++ {
			isMarked[frame] = true
		}
		isMarked[start-framesPerSample] = true // For first sample delta angles
        a_spotted_v := e.Attacker.HasSpotted(e.Player)
        v_spotted_a := e.Player.HasSpotted(e.Attacker)

		new := AttackTime{e.Attacker.UserID, e.Player.UserID, start, attackFrame, end, a_spotted_v, v_spotted_a}
		attackTimes = append(attackTimes, new)
	})

	// Track frames where a player fires a weapon
	var i int = 0
	p.RegisterEventHandler(func(e events.WeaponFire) {
		frame = p.CurrentFrame()
		// Include previous frames so that shot is not lost after sampling
		for i = 0; i < framesPerSample; i++ {
			fireFrames[FireFrameKey{e.Shooter.UserID, frame - i}] = true
		}
	})
	err = p.ParseToEnd()
    //fmt.Printf("Valid attacks: %d\n", attackCount)

	// Second pass.

	// Extract player data from marked frames
	f, err = os.Open(source + name)
	p = dem.NewParser(f)
    var ok bool

	for ok = true; ok; ok, err = p.ParseNextFrame() {
		checkError(err)
		frame = p.CurrentFrame()
		if !isMarked[frame] {
			continue
		}

		var players = map[int]PlayerData{}
		gs := p.GameState()

		for _, player := range gs.Participants().Playing() {
		    hasSpotted := gs.Participants().SpottedBy(player)
			players[player.UserID] = extractPlayerData(hasSpotted,frame, player, fireFrames)
		}
		markedFrameData[frame] = players
	}

	// Extract each attack's AttackData, and add it to modelData
	for _, attack := range attackTimes {
		//weapon := markedFrameData[attack.attackFrame][attack.attacker].weapon
		attackData := AttackData{}

		prevFrame := attack.startFrame - framesPerSample
		prevAttackerYaw := markedFrameData[prevFrame][attack.attacker].yaw
		prevAttackerPitch := markedFrameData[prevFrame][attack.attacker].pitch
		prevVictimYaw := markedFrameData[prevFrame][attack.victim].yaw
		prevVictimPitch := markedFrameData[prevFrame][attack.victim].pitch

        var attackerToVictim r3.Vector
        var attackerToVictimYaw float32
        var attackerToVictimPitch float32
        var sample int
        var attackerYaw float32
        var attackerPitch float32
        var victimToAttacker r3.Vector
        var victimToAttackerYaw float32
        var victimToAttackerPitch float32
        var victimYaw float32
        var victimPitch float32

		for sample = 0; sample < samplesPerAttack; sample++ {
			frame = framesPerSample*sample + attack.startFrame
			attacker := markedFrameData[frame][attack.attacker]
			victim := markedFrameData[frame][attack.victim]

            victimYaw = victim.yaw
            victimPitch = victim.pitch
			attackerYaw = attacker.yaw
			attackerPitch = attacker.pitch

			attackData.victimDeltaYaw[sample] = normalizeAngle(victimYaw - prevVictimYaw)
			attackData.victimDeltaPitch[sample] = victimPitch - prevVictimPitch
			attackData.attackerDeltaYaw[sample] = normalizeAngle(attackerYaw - prevAttackerYaw)
			attackData.attackerDeltaPitch[sample] = attackerPitch - prevAttackerPitch

			prevAttackerYaw = attackerYaw
			prevAttackerPitch = attackerPitch
			prevVictimYaw = victimYaw
			prevVictimPitch = victimPitch

			attackerToVictim = victim.position.Sub(attacker.position)
			attackData.attackerToVictimVector[sample] = attackerToVictim.Normalize()
            victimToAttacker = attacker.position.Sub(victim.position)
			attackData.attackerToVictimVector[sample] = victimToAttacker.Normalize()

			dXa := attackerToVictim.X
			dYa := attackerToVictim.Y
			dZa := attackerToVictim.Z

			dXv := victimToAttacker.X
			dYv := victimToAttacker.Y
			dZv := victimToAttacker.Z

			attackerToVictimYaw = 180 / math.Pi * float32(math.Atan2(dYa, dXa))
			attackerToVictimPitch = 180 / math.Pi * float32(math.Atan2(math.Sqrt(dXa*dXa+dYa*dYa),dZa))

			victimToAttackerYaw = 180 / math.Pi * float32(math.Atan2(dYv, dXv))
			victimToAttackerPitch = 180 / math.Pi * float32(math.Atan2(math.Sqrt(dXv*dXv+dYv*dYv),dZv))

			// Smallest angle between attackerToVictimYaw and attackerYaw
			attackData.crosshairToVictimYaw[sample] = normalizeAngle(attackerToVictimYaw - attackerYaw)
			attackData.crosshairToVictimPitch[sample] =	attackerToVictimPitch - attackerPitch

            attackData.crosshairToAttackerYaw[sample] = normalizeAngle(victimToAttackerYaw - victimYaw)
			attackData.crosshairToAttackerPitch[sample] =	victimToAttackerPitch - victimPitch

			attackData.victimDistance[sample] = float32(attackerToVictim.Norm())

			attackData.attackerCrouching[sample] = attacker.crouching
			attackData.victimCrouching[sample] = victim.crouching
			attackData.attackerFiring[sample] = attacker.firing
            attackData.victimFiring[sample] = victim.firing

			attackData.attackerHealth[sample] = attacker.health
			attackData.victimHealth[sample] = victim.health

			attackerYaw64 := float64(math.Pi / 180 * attackerYaw)
			attackerPitch64 := float64(math.Pi / 180 * attackerPitch)
			attackData.attackerViewVector[sample] = r3.Vector{
				math.Cos(attackerYaw64) * math.Sin(attackerPitch64),
				math.Sin(attackerYaw64) * math.Sin(attackerPitch64),
				math.Cos(attackerPitch64)}

			attackData.attackerX[sample] = float32(attacker.position.X)
			attackData.attackerY[sample] = float32(attacker.position.Y)
			attackData.attackerZ[sample] = float32(attacker.position.Z)

			attackData.victimX[sample] = float32(victim.position.X)
			attackData.victimY[sample] = float32(victim.position.Y)
			attackData.victimZ[sample] = float32(victim.position.Z)

            attackData.attackerxv[sample] = attacker.xv
            attackData.attackeryv[sample] = attacker.yv
            attackData.attackerzv[sample] = attacker.zv

            attackData.victimxv[sample] = victim.xv
            attackData.victimyv[sample] = victim.yv
            attackData.victimzv[sample] = victim.zv

            attackData.v_spotted_a[sample] = attack.v_spotted_a
            attackData.a_spotted_v[sample] = attack.a_spotted_v

            attackData.weapon[sample] = attacker.weapon
			attackData.flashleft[sample]  = attacker.flashleft

			attackData.name[sample]  = attacker.name
			attackData.id[sample] = attacker.id
			attackData.frame[sample] = frame
			attackData.IsScoped[sample] = attacker.IsScoped
            attackData.Ping[sample] = attacker.Ping
            attackData.nSpotted[sample] = victim.nSpotted
            attackData.AmmoLeft[sample] = attacker.weapon
            attackData.enemyPing[sample] = victim.Ping
		}
		// A player teleported. Throw away the data.
		modelData = append(modelData, attackData)
	}
	f.Close()
}

func extractPlayerData(
    spotted []*common.Player,
	frame int,
	player *common.Player,
	fireFrames map[FireFrameKey]bool) PlayerData{
	fixedPitch := float32(math.Mod(float64(player.ViewDirectionY())+90,180))
    ogViewDirectionX := player.ViewDirectionX()
    ogViewDirectionY := player.ViewDirectionY()

    var vel r3.Vector
    var xv float64
    var yv float64
    var zv float64

    vel = player.Velocity()
    xv = vel.X
    yv = vel.Y
    zv = vel.Z

    activeweapon := player.ActiveWeapon()
    var ammoleft int = -2
    var weapon int = -2
    // ActiveWeapon might return nil
    if activeweapon == nil{
        ammoleft = -2
        weapon = -2
    }else{
        ammoleft = activeweapon.AmmoInMagazine()
        weapon = int(activeweapon.Type)
    }

	return PlayerData{
	    ogViewDirectionX,
	    ogViewDirectionY,
	    ammoleft,
	    len(spotted),
	    player.Score(),
	    player.Ping(),
	    player.IsScoped(),
		player.FlashDuration,
		player.SteamID64,
		player.Name,
		xv,
		yv,
		zv,
		weapon,
		player.LastAlivePosition,
		player.ViewDirectionX(),
		fixedPitch,
		player.IsDucking(),
		fireFrames[FireFrameKey{player.UserID, frame}],
		player.Health()}
}



func csvExport() error {
    dataDest := "./data/data.csv"
	file, err := os.OpenFile(dataDest, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	writer := csv.NewWriter(file)

	for _, attackData := range modelData {
		err := writer.Write(attackToString(attackData))
		if err != nil {
			return err
		}
	}
	writer.Flush()
	file.Close()
	return nil
}


func attackToString(data AttackData) []string {
	var out []string

	for i := 0; i < samplesPerAttack; i++ {
		out = append(out, fmt.Sprintf(data.name[i]))

	    out = append(out, fmt.Sprintf("%.4f", data.ViewDirectionX[i]))
	    out = append(out, fmt.Sprintf("%.4f", data.ViewDirectionY[i]))

	    out = append(out, fmt.Sprintf("%d", data.AmmoLeft[i]))
	    out = append(out, fmt.Sprintf("%d", data.nSpotted[i]))
	    out = append(out, fmt.Sprintf("%d", data.Ping[i]))
	    out = append(out, fmt.Sprintf("%d", data.enemyPing[i]))
	    out = append(out, fmt.Sprintf("%.4f", data.flashleft[i]))
	    out = append(out, fmt.Sprintf("%d", data.weapon[i]))

        out = append(out, fmt.Sprintf("%.4f", data.attackerxv[i])) // v stands for velocity
        out = append(out, fmt.Sprintf("%.4f", data.attackeryv[i]))
        out = append(out, fmt.Sprintf("%.4f", data.attackerzv[i]))

        out = append(out, fmt.Sprintf("%.4f", data.victimxv[i]))
        out = append(out, fmt.Sprintf("%.4f", data.victimyv[i]))
        out = append(out, fmt.Sprintf("%.4f", data.victimzv[i]))

	    out = append(out, fmt.Sprintf("%.4f", data.victimDistance[i]))
        out = append(out, fmt.Sprintf("%d", int64(data.id[i])))
        out = append(out, fmt.Sprintf("%d", data.frame[i]))

		out = append(out, fmt.Sprintf("%.4f", data.attackerDeltaYaw[i]))
		out = append(out, fmt.Sprintf("%.4f", data.attackerDeltaPitch[i]))
		out = append(out, fmt.Sprintf("%.4f", data.crosshairToVictimYaw[i]))
		out = append(out, fmt.Sprintf("%.4f", data.crosshairToVictimPitch[i]))

        out = append(out, fmt.Sprintf("%.4f", data.victimDeltaYaw[i]))
		out = append(out, fmt.Sprintf("%.4f", data.victimDeltaPitch[i]))
		out = append(out, fmt.Sprintf("%.4f", data.crosshairToAttackerYaw[i]))
		out = append(out, fmt.Sprintf("%.4f", data.crosshairToAttackerPitch[i]))

        if data.IsScoped[i] {
                out = append(out,  fmt.Sprintf("%d", 1))
            } else {
                out = append(out, fmt.Sprintf("%d", 0))
            }

        if data.v_spotted_a[i] {
                out = append(out,  fmt.Sprintf("%d", 1))
            } else {
                out = append(out, fmt.Sprintf("%d", 0))
            }

        if data.a_spotted_v[i] {
                out = append(out,  fmt.Sprintf("%d", 1))
            } else {
                out = append(out, fmt.Sprintf("%d", 0))
            }

        if data.attackerFiring[i] {
                out = append(out,  fmt.Sprintf("%d", 1))
            } else {
                out = append(out, fmt.Sprintf("%d", 0))
            }
        if data.victimFiring[i] {
                out = append(out,  fmt.Sprintf("%d", 1))
            } else {
                out = append(out, fmt.Sprintf("%d", 0))
            }
        }

	return out
}

// Returns a mod b, keeping the sign of b
func divisorSignMod(a float64, b float64) float64 {
	return math.Mod(math.Mod(a, b)+b, b)
}

// Normalize an angle to be between -180 and 180
func normalizeAngle(a float32) float32 {
	return float32(-180 + divisorSignMod(float64(a)+180, 360))
}

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}
