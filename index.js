require('dotenv').config();
const path = require('path');
const fs = require('fs');
const fse = require('fs-extra');
const { pipeline } = require('stream');
const { Client, GatewayIntentBits, Events } = require('discord.js');
const { joinVoiceChannel, EndBehaviorType, getVoiceConnection } = require('@discordjs/voice');
const prism = require('prism-media');
const wav = require('wav');

// === Env ===
const TOKEN = process.env.DISCORD_TOKEN;
const RECORDINGS_DIR = process.env.RECORDINGS_DIR || path.join(process.cwd(), 'recordings');
const SESSION_PREFIX = process.env.SESSION_PREFIX || 'session';
if (!TOKEN) {
  console.error('Missing DISCORD_TOKEN in .env');
  process.exit(1);
}

// === Client ===
const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
});

// Stan sesji per-guild
const sessions = new Map(); // guildId -> { baseDir, channelId, startISO, counters, active }

// Rejestracja slashów globalnie
async function ensureCommands() {
  const commands = [
    {
      name: 'record',
      description: 'Record the current voice channel',
      options: [
        { name: 'start', description: 'Start recording', type: 1 },
        { name: 'stop', description: 'Stop recording', type: 1 },
      ],
    },
  ];
  await client.application.commands.set(commands);
}

client.once(Events.ClientReady, async () => {
  console.log(`Logged in as ${client.user.tag}`);
  try {
    await ensureCommands();
    console.log('Slash commands ready.');
  } catch (e) {
    console.error('Cmds error:', e);
  }
});

client.on(Events.InteractionCreate, async (interaction) => {
  try {
    if (!interaction.isChatInputCommand()) return;
    if (interaction.commandName !== 'record') return;

    const sub = interaction.options.getSubcommand();
    const guildId = interaction.guildId;

    if (sub === 'start') {
      const member = await interaction.guild.members.fetch(interaction.user.id);
      const voice = member.voice?.channel;
      if (!voice) {
        return interaction.reply({ content: 'Wejdź najpierw na kanał głosowy.', flags: 64 });
      }

      // Start sesji
      const ts = new Date().toISOString().replace(/[:.]/g, '-');
      const baseDir = path.join(RECORDINGS_DIR, `${SESSION_PREFIX}-${ts}-${voice.id}`);
      const rawDir = path.join(baseDir, 'raw');
      await fse.ensureDir(rawDir);

      const manifest = {
        guildId,
        channelId: voice.id,
        channelName: voice.name,
        startISO: new Date().toISOString(),
        bot: client.user.tag,
        versions: { node: process.version },
      };
      await fse.writeJson(path.join(baseDir, 'manifest.json'), manifest, { spaces: 2 });

      const connection = joinVoiceChannel({
        channelId: voice.id,
        guildId: voice.guild.id,
        adapterCreator: voice.guild.voiceAdapterCreator,
        selfDeaf: false,
        selfMute: true,
      });

      // --- diagnostyka połączenia i mowy
      console.log('[voice] joined VC:', voice.name, 'guild:', voice.guild.id);
      connection.on('stateChange', (oldS, newS) => {
        console.log('[voice] conn state:', oldS.status, '->', newS.status);
      });

      const receiver = connection.receiver;
      const counters = new Map(); // userId -> segment index

      receiver.speaking.on('start', (userId) => {
        console.log('[voice] speaking START:', userId);
      });
      receiver.speaking.on('end', (userId) => {
        console.log('[voice] speaking END:', userId);
      });

      receiver.speaking.on('start', async (userId) => {
        try {
          const user = await client.users.fetch(userId).catch(() => null);
          const label = user ? `${user.username}_${user.id}` : `user_${userId}`;
          const idx = (counters.get(userId) || 0) + 1;
          counters.set(userId, idx);

          const filename = `${label}_seg${String(idx).padStart(3, '0')}.wav`;
          const filePath = path.join(rawDir, filename);

          const opusStream = receiver.subscribe(userId, {
            end: { behavior: EndBehaviorType.AfterSilence, duration: 1500 },
          });

          const decoder = new prism.opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 });
          const wavWriter = new wav.Writer({ channels: 2, sampleRate: 48000, bitDepth: 16 });
          const out = fs.createWriteStream(filePath);

          // --- BEZPIECZNY ZAPIS: usuwaj puste pliki (0 KB)
          let bytesWritten = 0;
          const origWrite = out.write.bind(out);
          out.write = (chunk, ...args) => {
            if (Buffer.isBuffer(chunk)) bytesWritten += chunk.length;
            return origWrite(chunk, ...args);
          };
          out.on('finish', () => {
            if (bytesWritten === 0) {
              fs.unlink(filePath, () => {});
              console.log(`[voice] empty chunk, removed: ${filePath}`);
            }
          });

          pipeline(opusStream, decoder, wavWriter, out, (err) => {
            if (err) console.error('Pipeline error:', err);
            else if (bytesWritten > 0) console.log(`Saved: ${filePath} (${(bytesWritten / 1024) | 0} KB)`);
          });
        } catch (e) {
          console.error('speaking start err:', e);
        }
      });

      sessions.set(guildId, {
        baseDir,
        channelId: voice.id,
        startISO: new Date().toISOString(),
        counters,
        active: true,
      });

      await interaction.reply({
        content: `Nagrywam kanał **${voice.name}**. Pliki: \`${baseDir}\``,
        flags: 64,
      });
    }

    if (sub === 'stop') {
      const sess = sessions.get(guildId);
      const conn = getVoiceConnection(guildId);

      if (!sess || !conn) {
        return interaction.reply({ content: 'Brak aktywnej sesji na tym serwerze.', flags: 64 });
      }

      sess.active = false;
      conn.destroy();

      // dopisz stop do manifestu
      const manifestPath = path.join(sess.baseDir, 'manifest.json');
      try {
        const man = await fse.readJson(manifestPath);
        man.stopISO = new Date().toISOString();
        await fse.writeJson(manifestPath, man, { spaces: 2 });
      } catch (_) {}

      await interaction.reply({ content: `Zatrzymane. Pliki w: \`${sess.baseDir}\``, flags: 64 });
    }
  } catch (e) {
    console.error('Interaction error:', e);
    if (interaction.isRepliable()) {
      interaction.reply({ content: 'Ups, błąd. Sprawdź logi kontenera.', flags: 64 }).catch(() => {});
    }
  }
});

client.login(TOKEN);
