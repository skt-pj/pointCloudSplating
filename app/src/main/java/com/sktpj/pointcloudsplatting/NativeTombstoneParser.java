package com.sktpj.pointcloudsplatting;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Minimal parser for Android debuggerd tombstone protobuf fields needed by PCS diagnostics. */
final class NativeTombstoneParser {
    private static final int MAX_TRACE_BYTES = 1_000_000;
    private static final int MAX_FRAMES = 40;

    private NativeTombstoneParser() {}

    static String parse(InputStream input) throws IOException {
        byte[] bytes = readLimited(input);
        Tombstone tombstone = parseTombstone(new Cursor(bytes));
        StringBuilder out = new StringBuilder();
        out.append("bytes=").append(bytes.length)
                .append(" pid=").append(tombstone.pid)
                .append(" tid=").append(tombstone.tid);
        if (tombstone.signalName != null && !tombstone.signalName.isEmpty()) {
            out.append(" signal=").append(tombstone.signalName);
            if (tombstone.signalNumber != 0) out.append('(').append(tombstone.signalNumber).append(')');
        } else if (tombstone.signalNumber != 0) {
            out.append(" signal=").append(tombstone.signalNumber);
        }
        if (tombstone.codeName != null && !tombstone.codeName.isEmpty()) {
            out.append(" code=").append(tombstone.codeName);
            if (tombstone.code != 0) out.append('(').append(tombstone.code).append(')');
        } else if (tombstone.code != 0) {
            out.append(" code=").append(tombstone.code);
        }
        if (tombstone.abortMessage != null && !tombstone.abortMessage.isEmpty()) {
            out.append(" abort=").append(sanitize(tombstone.abortMessage));
        }

        ThreadInfo crashThread = null;
        for (ThreadInfo thread : tombstone.threads) {
            if (thread.mapKey == tombstone.tid || thread.id == tombstone.tid) {
                crashThread = thread;
                break;
            }
        }
        if (crashThread == null && !tombstone.threads.isEmpty()) {
            crashThread = tombstone.threads.get(0);
        }
        if (crashThread != null) {
            out.append("\nthread name=").append(sanitize(crashThread.name == null ? "" : crashThread.name))
                    .append(" id=").append(crashThread.id);
            int count = Math.min(MAX_FRAMES, crashThread.frames.size());
            for (int i = 0; i < count; ++i) {
                Frame frame = crashThread.frames.get(i);
                out.append("\n#").append(String.format(Locale.US, "%02d", i))
                        .append(" rel_pc=0x").append(Long.toUnsignedString(frame.relPc, 16));
                if (frame.fileName != null && !frame.fileName.isEmpty()) {
                    out.append(' ').append(frame.fileName);
                }
                if (frame.functionName != null && !frame.functionName.isEmpty()) {
                    out.append(" (").append(frame.functionName);
                    if (frame.functionOffset != 0) {
                        out.append("+0x").append(Long.toUnsignedString(frame.functionOffset, 16));
                    }
                    out.append(')');
                }
                if (frame.buildId != null && !frame.buildId.isEmpty()) {
                    out.append(" buildId=").append(frame.buildId);
                }
            }
        }
        return out.toString();
    }

    private static byte[] readLimited(InputStream input) throws IOException {
        if (input == null) throw new IOException("native tombstone stream unavailable");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[32 * 1024];
        int total = 0;
        while (true) {
            int n = input.read(buffer);
            if (n < 0) break;
            total += n;
            if (total > MAX_TRACE_BYTES) throw new IOException("native tombstone exceeds 1 MB");
            out.write(buffer, 0, n);
        }
        return out.toByteArray();
    }

    private static Tombstone parseTombstone(Cursor c) throws IOException {
        Tombstone out = new Tombstone();
        while (!c.eof()) {
            long tag = c.readVarint();
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            switch (field) {
                case 5:
                    requireWire(field, wire, 0);
                    out.pid = (int) c.readVarint();
                    break;
                case 6:
                    requireWire(field, wire, 0);
                    out.tid = (int) c.readVarint();
                    break;
                case 10:
                    requireWire(field, wire, 2);
                    parseSignal(new Cursor(c.readBytes()), out);
                    break;
                case 14:
                    requireWire(field, wire, 2);
                    out.abortMessage = c.readString();
                    break;
                case 16:
                    requireWire(field, wire, 2);
                    out.threads.add(parseThreadMapEntry(new Cursor(c.readBytes())));
                    break;
                default:
                    c.skipField(wire);
                    break;
            }
        }
        return out;
    }

    private static void parseSignal(Cursor c, Tombstone out) throws IOException {
        while (!c.eof()) {
            long tag = c.readVarint();
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            switch (field) {
                case 1:
                    requireWire(field, wire, 0);
                    out.signalNumber = (int) c.readVarint();
                    break;
                case 2:
                    requireWire(field, wire, 2);
                    out.signalName = c.readString();
                    break;
                case 3:
                    requireWire(field, wire, 0);
                    out.code = (int) c.readVarint();
                    break;
                case 4:
                    requireWire(field, wire, 2);
                    out.codeName = c.readString();
                    break;
                default:
                    c.skipField(wire);
                    break;
            }
        }
    }

    private static ThreadInfo parseThreadMapEntry(Cursor c) throws IOException {
        int key = 0;
        ThreadInfo thread = null;
        while (!c.eof()) {
            long tag = c.readVarint();
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            if (field == 1) {
                requireWire(field, wire, 0);
                key = (int) c.readVarint();
            } else if (field == 2) {
                requireWire(field, wire, 2);
                thread = parseThread(new Cursor(c.readBytes()));
            } else {
                c.skipField(wire);
            }
        }
        if (thread == null) thread = new ThreadInfo();
        thread.mapKey = key;
        return thread;
    }

    private static ThreadInfo parseThread(Cursor c) throws IOException {
        ThreadInfo out = new ThreadInfo();
        while (!c.eof()) {
            long tag = c.readVarint();
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            switch (field) {
                case 1:
                    requireWire(field, wire, 0);
                    out.id = (int) c.readVarint();
                    break;
                case 2:
                    requireWire(field, wire, 2);
                    out.name = c.readString();
                    break;
                case 4:
                    requireWire(field, wire, 2);
                    out.frames.add(parseFrame(new Cursor(c.readBytes())));
                    break;
                default:
                    c.skipField(wire);
                    break;
            }
        }
        return out;
    }

    private static Frame parseFrame(Cursor c) throws IOException {
        Frame out = new Frame();
        while (!c.eof()) {
            long tag = c.readVarint();
            int field = (int) (tag >>> 3);
            int wire = (int) (tag & 7);
            switch (field) {
                case 1:
                    requireWire(field, wire, 0);
                    out.relPc = c.readVarint();
                    break;
                case 4:
                    requireWire(field, wire, 2);
                    out.functionName = c.readString();
                    break;
                case 5:
                    requireWire(field, wire, 0);
                    out.functionOffset = c.readVarint();
                    break;
                case 6:
                    requireWire(field, wire, 2);
                    out.fileName = c.readString();
                    break;
                case 8:
                    requireWire(field, wire, 2);
                    out.buildId = c.readString();
                    break;
                default:
                    c.skipField(wire);
                    break;
            }
        }
        return out;
    }

    private static void requireWire(int field, int actual, int expected) throws IOException {
        if (actual != expected) {
            throw new IOException("unexpected protobuf wire type field=" + field
                    + " actual=" + actual + " expected=" + expected);
        }
    }

    private static String sanitize(String value) {
        return value.replace('\n', ' ').replace('\r', ' ');
    }

    private static final class Tombstone {
        int pid;
        int tid;
        int signalNumber;
        int code;
        String signalName;
        String codeName;
        String abortMessage;
        final List<ThreadInfo> threads = new ArrayList<>();
    }

    private static final class ThreadInfo {
        int mapKey;
        int id;
        String name;
        final List<Frame> frames = new ArrayList<>();
    }

    private static final class Frame {
        long relPc;
        long functionOffset;
        String functionName;
        String fileName;
        String buildId;
    }

    private static final class Cursor {
        final byte[] data;
        int pos;

        Cursor(byte[] data) {
            this.data = data == null ? new byte[0] : data;
        }

        boolean eof() {
            return pos >= data.length;
        }

        long readVarint() throws IOException {
            long result = 0;
            for (int shift = 0; shift < 64; shift += 7) {
                if (pos >= data.length) throw new IOException("truncated protobuf varint");
                int b = data[pos++] & 0xff;
                result |= (long) (b & 0x7f) << shift;
                if ((b & 0x80) == 0) return result;
            }
            throw new IOException("protobuf varint too long");
        }

        byte[] readBytes() throws IOException {
            long rawLength = readVarint();
            if (rawLength < 0 || rawLength > Integer.MAX_VALUE) {
                throw new IOException("invalid protobuf length=" + rawLength);
            }
            int length = (int) rawLength;
            if (length > data.length - pos) throw new IOException("truncated protobuf bytes");
            byte[] out = new byte[length];
            System.arraycopy(data, pos, out, 0, length);
            pos += length;
            return out;
        }

        String readString() throws IOException {
            return new String(readBytes(), StandardCharsets.UTF_8);
        }

        void skipField(int wire) throws IOException {
            switch (wire) {
                case 0:
                    readVarint();
                    return;
                case 1:
                    skip(8);
                    return;
                case 2:
                    long length = readVarint();
                    if (length < 0 || length > Integer.MAX_VALUE) {
                        throw new IOException("invalid protobuf skip length=" + length);
                    }
                    skip((int) length);
                    return;
                case 5:
                    skip(4);
                    return;
                default:
                    throw new IOException("unsupported protobuf wire type=" + wire);
            }
        }

        void skip(int count) throws IOException {
            if (count < 0 || count > data.length - pos) throw new IOException("truncated protobuf field");
            pos += count;
        }
    }
}
