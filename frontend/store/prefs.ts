import { create } from "zustand";
import { persist } from "zustand/middleware";

interface PrefsState {
  followedPlayers: string[];
  followedTeams:   string[];
  defaultStat:     string;
  defaultSort:     "ev" | "hit_rate";

  followPlayer:   (name: string) => void;
  unfollowPlayer: (name: string) => void;
  followTeam:     (abbrev: string) => void;
  unfollowTeam:   (abbrev: string) => void;
  setDefaultStat: (stat: string) => void;
  setDefaultSort: (sort: "ev" | "hit_rate") => void;
}

export const usePrefs = create<PrefsState>()(
  persist(
    (set, get) => ({
      followedPlayers: [],
      followedTeams:   [],
      defaultStat:     "",
      defaultSort:     "ev",

      followPlayer: (name) =>
        set({ followedPlayers: [...new Set([...get().followedPlayers, name])] }),
      unfollowPlayer: (name) =>
        set({ followedPlayers: get().followedPlayers.filter((p) => p !== name) }),

      followTeam: (abbrev) =>
        set({ followedTeams: [...new Set([...get().followedTeams, abbrev])] }),
      unfollowTeam: (abbrev) =>
        set({ followedTeams: get().followedTeams.filter((t) => t !== abbrev) }),

      setDefaultStat: (stat) => set({ defaultStat: stat }),
      setDefaultSort: (sort) => set({ defaultSort: sort }),
    }),
    { name: "nba-prefs" }
  )
);
